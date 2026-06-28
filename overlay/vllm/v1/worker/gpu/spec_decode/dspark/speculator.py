# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark parallel-drafting speculator for the V2 model runner.

DSpark uses the same context-K/V precompute shape as DFlash, but its query
block is [anchor_token, noise, ...] with exactly N query positions. Every
query position is sampled, then the Markov head conditions step i on the
previous draft token.
"""

import os
from collections import deque
from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


def _parse_verify_len_buckets(
    raw: str | None,
    max_verify_len: int,
) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    buckets: set[int] = {max_verify_len}
    for part in raw.replace(";", ",").replace(" ", ",").split(","):
        if not part:
            continue
        try:
            bucket = int(part)
        except ValueError:
            logger.warning_once(
                "Ignoring invalid DSPARK_VERIFY_LEN_BUCKETS entry %r in %r.",
                part,
                raw,
            )
            continue
        if 1 <= bucket <= max_verify_len:
            buckets.add(bucket)
        else:
            logger.warning_once(
                "Ignoring out-of-range DSpark verify bucket %s; expected 1..%s.",
                bucket,
                max_verify_len,
            )
    return tuple(sorted(buckets))


class DSparkSpeculator(DFlashSpeculator):
    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)

        # DFlash uses a bonus token plus N masks. DSpark samples the anchor
        # position itself, so the draft query block has exactly N tokens.
        self.num_query_per_req = self.num_speculative_steps
        max_query_tokens = self.max_num_reqs * self.num_query_per_req
        assert max_query_tokens <= self.max_num_tokens, (
            "max_num_batched_tokens is too small for the DSpark draft block "
            f"({max_query_tokens} > {self.max_num_tokens})."
        )
        dspark_hidden_size = self.draft_model_config.get_hidden_size()
        if self.hidden_states.shape[1] != dspark_hidden_size:
            self.hidden_states = torch.zeros(
                self.max_num_tokens,
                dspark_hidden_size,
                dtype=self.dtype,
                device=device,
            )
        self.confidence_logits = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.float32,
            device=device,
        )
        self.draft_kv_cache_group_ids: list[int] = []
        self.context_slot_mappings: torch.Tensor | None = None
        self.context_commit_mask: torch.Tensor | None = None
        self._committed_prefix_kv = os.getenv("DSPARK_COMMITTED_PREFIX_KV", "0") == "1"
        self._confidence_schedule = bool(
            vllm_config.speculative_config.dspark_confidence_schedule
        )
        self._last_scheduled_draft_lengths: torch.Tensor | None = None
        self._survival_history: deque[torch.Tensor] = deque(maxlen=2)
        self._logged_confidence_shape = False
        sts_temperatures = vllm_config.speculative_config.dspark_sts_temperatures
        if sts_temperatures is None:
            sts_temperatures = [
                1.15 + 0.15 * idx for idx in range(self.num_speculative_steps)
            ]
            if self._confidence_schedule:
                logger.warning_once(
                    "DSpark confidence scheduler enabled without a shipped STS "
                    "calibration table; using conservative default per-position "
                    "temperatures: %s",
                    [round(t, 4) for t in sts_temperatures],
                )
        self._sts_temperatures = torch.tensor(
            sts_temperatures,
            dtype=torch.float32,
            device=device,
        )
        self._scheduler_knee_tokens = float(
            vllm_config.speculative_config.dspark_scheduler_knee_tokens
        )
        self._scheduler_cost_exponent = float(
            vllm_config.speculative_config.dspark_scheduler_cost_exponent
        )
        self._verify_len_buckets = _parse_verify_len_buckets(
            os.getenv("DSPARK_VERIFY_LEN_BUCKETS"),
            self.num_speculative_steps + 1,
        )
        self._verify_len_bucket_tensor = (
            torch.tensor(
                self._verify_len_buckets,
                dtype=torch.int32,
                device=device,
            )
            if self._verify_len_buckets is not None
            else None
        )
        if self._confidence_schedule and self._verify_len_buckets is not None:
            logger.info(
                "DSpark confidence scheduler using verify-length CUDA graph "
                "buckets: %s",
                list(self._verify_len_buckets),
            )
        if self._committed_prefix_kv:
            logger.info(
                "DSpark committed-prefix context KV precompute is enabled."
            )

    def take_last_scheduled_draft_lengths(self) -> torch.Tensor | None:
        return self._last_scheduled_draft_lengths

    def _bucket_draft_lengths(self, draft_lengths: torch.Tensor) -> torch.Tensor:
        buckets = self._verify_len_bucket_tensor
        if buckets is None:
            return draft_lengths
        verify_lens = draft_lengths.to(torch.int32) + 1
        bucket_idx = torch.bucketize(verify_lens, buckets, right=True) - 1
        bucket_idx = torch.clamp(bucket_idx, min=0, max=buckets.numel() - 1)
        return (buckets[bucket_idx] - 1).to(torch.int32)

    def _calibrated_survival_probs(
        self,
        confidence_logits: torch.Tensor,
    ) -> torch.Tensor:
        temperatures = self._sts_temperatures[: confidence_logits.shape[1]]
        conditional = torch.sigmoid(
            confidence_logits.float() / temperatures.view(1, -1)
        )
        conditional = conditional.clamp_(1e-6, 1.0)
        return conditional.cumprod(dim=1)

    def _normalized_sps(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        batch_tokens = batch_tokens.to(torch.float32)
        load = batch_tokens / self._scheduler_knee_tokens
        return torch.reciprocal(1.0 + torch.pow(load, self._scheduler_cost_exponent))

    def _schedule_draft_lengths(
        self,
        confidence_logits: torch.Tensor,
    ) -> torch.Tensor:
        survival_probs = self._calibrated_survival_probs(confidence_logits)
        num_reqs, max_draft = survival_probs.shape
        if not self._logged_confidence_shape:
            try:
                means = survival_probs.mean(dim=0).detach().cpu().tolist()
                logger.info(
                    "DSpark confidence head online: logits_shape=%s "
                    "mean_calibrated_prefix_survival=%s",
                    tuple(confidence_logits.shape),
                    [round(float(v), 4) for v in means],
                )
            except Exception:
                logger.exception("Failed to log DSpark confidence head summary")
            self._logged_confidence_shape = True

        if num_reqs <= 1:
            self._survival_history.append(survival_probs.detach())
            return torch.full(
                (num_reqs,),
                max_draft,
                dtype=torch.int32,
                device=confidence_logits.device,
            )

        if len(self._survival_history) < 2:
            self._survival_history.append(survival_probs.detach())
            return torch.full(
                (num_reqs,),
                max_draft,
                dtype=torch.int32,
                device=confidence_logits.device,
            )

        if num_reqs <= 4:
            # At low concurrency the target pass is already efficient enough
            # that shortening loses more accepted tokens than it saves. Keep
            # the full verifier bucket and let higher-concurrency batches trade
            # acceptance for smaller target graphs.
            self._survival_history.append(survival_probs.detach())
            return torch.full(
                (num_reqs,),
                max_draft,
                dtype=torch.int32,
                device=confidence_logits.device,
            )
        else:
            min_prefix = max(max_draft - 3, 1)
            threshold = 0.38
        per_req_lengths = (survival_probs >= threshold).sum(dim=1).to(torch.int32)
        per_req_lengths = torch.clamp(per_req_lengths, min=min_prefix, max=max_draft)
        lengths = per_req_lengths.min().expand(num_reqs).contiguous()
        lengths = self._bucket_draft_lengths(lengths)
        self._survival_history.append(survival_probs.detach())
        return lengths

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        DraftModelSpeculator.set_attn(
            self, model_state, kv_cache_config, block_tables
        )

        draft_groups = [gid for gid, g in enumerate(self.attn_groups) if g]
        if not draft_groups:
            raise RuntimeError("DSpark did not find any draft attention KV groups.")

        draft_block_sizes = {self.block_tables.block_sizes[gid] for gid in draft_groups}
        if len(draft_block_sizes) != 1:
            raise RuntimeError(
                "DSpark requires all draft attention KV groups to use the same "
                f"block size; got {sorted(draft_block_sizes)}."
            )

        self.draft_kv_cache_group_ids = draft_groups
        self.draft_kv_cache_group_id = draft_groups[0]
        self.draft_block_size = self.block_tables.block_sizes[
            self.draft_kv_cache_group_id
        ]
        self.context_slot_mappings = torch.zeros(
            len(draft_groups),
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        self.context_commit_mask = torch.zeros(
            self.max_num_tokens,
            dtype=torch.bool,
            device=self.device,
        )

    def _context_slot_mappings_by_layer(
        self, num_target_tokens: int
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        assert self.context_slot_mappings is not None
        if len(self.draft_kv_cache_group_ids) == 1:
            return self.context_slot_mappings[0, :num_target_tokens]

        context_slot_mappings: dict[str, torch.Tensor] = {}
        for group_idx, kv_cache_group_id in enumerate(self.draft_kv_cache_group_ids):
            slot_mapping = self.context_slot_mappings[group_idx, :num_target_tokens]
            for layer_name in self.kv_cache_config.kv_cache_groups[
                kv_cache_group_id
            ].layer_names:
                if layer_name in self.draft_attn_layer_names:
                    context_slot_mappings[layer_name] = slot_mapping
        return context_slot_mappings

    def capture(self, attn_states: dict | None = None) -> None:
        logger.info("Capturing model for DSpark speculator...")
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.zero_()
        assert self.query_cudagraph_manager is not None
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            progress_bar_desc="Capturing dspark CUDA graphs",
        )

    def sample_dspark_draft(
        self,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = anchor_token_ids.shape[0]
        num_steps = hidden_states.shape[0] // batch_size
        base_logits = self.model.compute_logits(hidden_states)
        assert base_logits is not None
        hidden_states_by_step = hidden_states.view(batch_size, num_steps, -1)
        base_logits = base_logits.view(batch_size, num_steps, -1)
        positions = positions.view(batch_size, num_steps)
        idx_mapping = idx_mapping.view(batch_size, num_steps)
        draft_step = draft_step.view(batch_size, num_steps)

        prev_token_ids = anchor_token_ids.to(torch.int64)
        draft_token_ids = []
        confidence_logits = (
            [] if self._confidence_schedule and batch_size > 4 else None
        )
        final_layer = self.model.model.final_dspark_layer
        for step_idx in range(num_steps):
            logits_bias, markov_embed = final_layer.markov_head(prev_token_ids)
            if confidence_logits is not None:
                confidence_logits.append(
                    final_layer.confidence_head(
                        hidden_states_by_step[:, step_idx, :],
                        markov_embed,
                    ).float()
                )
            step_logits = base_logits[:, step_idx, :] + logits_bias
            if draft_logits is not None:
                next_token_ids = gumbel_sample(
                    step_logits,
                    idx_mapping[:, step_idx],
                    temperature,
                    seeds,
                    positions[:, step_idx] + 1,
                    apply_temperature=True,
                    output_processed_logits=draft_logits,
                    output_processed_logits_col=draft_step[:, step_idx],
                    use_fp64=self.use_fp64_gumbel,
                )
            else:
                next_token_ids = step_logits.argmax(dim=-1)
            draft_token_ids.append(next_token_ids)
            prev_token_ids = next_token_ids

        if confidence_logits is not None:
            self.confidence_logits[:batch_size, :num_steps].copy_(
                torch.stack(confidence_logits, dim=1).contiguous()
            )
        return torch.stack(draft_token_ids, dim=1)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        anchor_indices = (
            torch.arange(num_reqs, dtype=torch.int64, device=self.device)
            * self.num_query_per_req
        )
        anchor_token_ids = self.input_buffers.input_ids[anchor_indices]
        self.draft_tokens[:num_reqs] = self.sample_dspark_draft(
            sample_hidden_states,
            anchor_token_ids,
            self.sample_pos[:num_sample],
            self.sample_idx_mapping[:num_sample],
            self.temperature,
            self.seeds,
            self.sample_col[:num_sample],
            self.draft_logits,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del attn_metadata, slot_mappings, mm_inputs

        self._last_scheduled_draft_lengths = None
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        self.draft_max_seq_len = min(
            input_batch.max_seq_len_upper_bound + self.num_query_per_req,
            self.max_model_len,
        )

        if aux_hidden_states:
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        if dummy_run and skip_attn_for_dummy_run:
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_target_tokens],
                self.context_positions[:num_target_tokens],
            )
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]

        assert self.draft_kv_cache_group_ids
        assert self.context_slot_mappings is not None
        assert self.context_commit_mask is not None
        for group_idx, kv_cache_group_id in enumerate(self.draft_kv_cache_group_ids):
            prepare_dspark_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[kv_cache_group_id],
                self.context_positions,
                self.context_slot_mappings[group_idx],
                self.context_commit_mask,
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                self.block_tables.input_block_tables[kv_cache_group_id],
                self.draft_block_size,
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                self.max_num_tokens,
            )

        self.model.precompute_and_store_context_kv(
            self.hidden_states[:num_target_tokens],
            self.context_positions[:num_target_tokens],
            context_slot_mapping=(
                None
                if dummy_run
                else self._context_slot_mappings_by_layer(num_target_tokens)
            ),
            context_commit_mask=(
                None
                if dummy_run or not self._committed_prefix_kv
                else self.context_commit_mask[:num_target_tokens]
            ),
        )

        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_query_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens
        draft_attn_metadata = self._build_draft_attn_metadata(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            causal=self.dflash_causal,
        )
        draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(
                num_reqs_padded,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )

        if self._confidence_schedule and not (dummy_run or is_profile):
            if num_reqs <= 4:
                self._last_scheduled_draft_lengths = torch.full(
                    (num_reqs,),
                    self.num_speculative_steps,
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                self._last_scheduled_draft_lengths = self._schedule_draft_lengths(
                    self.confidence_logits[:num_reqs]
                )

        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_dspark_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_context_commit_mask_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_committed_ctx = valid_ctx_end - ctx_start

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        anchor_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        anchor_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_query = (j >= num_ctx) & (j < num_ctx + num_query_per_req)
    query_off = j - num_ctx

    # --- Context positions / slots ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_ctx,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)
    tl.store(
        out_context_commit_mask_ptr + ctx_start + j,
        j < num_committed_ctx,
        mask=is_ctx,
    )

    # --- Query positions / input_ids / slots ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_anchor = is_query & (query_off == 0)
    input_id = tl.where(is_anchor, anchor_token, parallel_drafting_token_id)

    q_block_num = query_pos // block_size
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_slot = q_block_id * block_size + (query_pos % block_size)

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    tl.store(out_query_positions_ptr + query_idx, query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # --- Sample indices / positions / idx_mapping (all DSpark query tokens) ---
    sample_idx = req_idx * num_speculative_steps + query_off
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_query)
    tl.store(out_sample_pos_ptr + sample_idx, query_pos, mask=is_query)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_query)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req)
        if req_idx == num_reqs - 1:
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, 0, mask=mask)
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def prepare_dspark_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    context_commit_mask: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: InputBatch,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    # [max_num_reqs, max_num_blocks]
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    block_size_tokens = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, block_size_tokens)
    _prepare_dspark_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        context_commit_mask,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=block_size_tokens,
    )
