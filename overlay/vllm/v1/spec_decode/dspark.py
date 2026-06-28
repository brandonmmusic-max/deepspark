# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections import deque
from typing import Any

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.utils import copy_and_expand_dspark_inputs_kernel

logger = init_logger(__name__)


class DSparkProposer(DFlashProposer):
    """DFlash-style proposer with DSpark's anchor-position draft sampling."""

    def __init__(
        self,
        vllm_config,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        self._dspark_anchor_token_ids: torch.Tensor | None = None
        self._confidence_schedule = bool(
            vllm_config.speculative_config.dspark_confidence_schedule
        )
        self._last_confidence_logits: torch.Tensor | None = None
        self._last_scheduled_draft_lengths: torch.Tensor | None = None
        self._survival_history: deque[torch.Tensor] = deque(maxlen=2)
        self._logged_confidence_shape = False
        self._committed_prefix_kv = os.getenv("DSPARK_COMMITTED_PREFIX_KV", "0") == "1"
        self._context_commit_mask_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.bool,
            device=device,
        )
        if self._committed_prefix_kv:
            logger.info(
                "DSpark committed-prefix context KV precompute is enabled."
            )

        sts_temperatures = vllm_config.speculative_config.dspark_sts_temperatures
        if sts_temperatures is None:
            sts_temperatures = [
                1.15 + 0.15 * idx for idx in range(self.num_speculative_tokens)
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

    def take_last_scheduled_draft_lengths(self) -> torch.Tensor | None:
        return self._last_scheduled_draft_lengths

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

        if num_reqs <= 2:
            min_prefix = max(max_draft - 1, 1)
            threshold = 0.18
        elif num_reqs <= 4:
            min_prefix = max(max_draft - 2, 1)
            threshold = 0.28
        else:
            min_prefix = max(max_draft - 3, 1)
            threshold = 0.38
        per_req_lengths = (survival_probs >= threshold).sum(dim=1).to(torch.int32)
        per_req_lengths = torch.clamp(per_req_lengths, min=min_prefix, max=max_draft)
        lengths = per_req_lengths.min().expand(num_reqs).contiguous()
        self._survival_history.append(survival_probs.detach())
        return lengths

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        self._dflash_num_context = num_context
        self._dflash_hidden_states = target_hidden_states
        self._dspark_anchor_token_ids = next_token_ids

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        block_size_tokens = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, block_size_tokens)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        self._ensure_slot_mapping_buffers()
        draft_kv_group_ids = self._draft_kv_gids()
        for kv_cache_gid in draft_kv_group_ids:
            context_slot_mapping_buffer, query_slot_mapping_buffer = (
                self._slot_mapping_buffers_by_gid[kv_cache_gid]
            )
            block_table = self._get_dflash_block_table(kv_cache_gid, cad)
            copy_and_expand_dspark_inputs_kernel[grid](
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=context_slot_mapping_buffer,
                out_context_commit_mask_ptr=self._context_commit_mask_buffer,
                out_query_slot_mapping_ptr=query_slot_mapping_buffer,
                out_token_indices_ptr=token_indices_to_sample,
                block_table_ptr=block_table,
                block_table_stride=block_table.stride(0),
                query_start_loc_ptr=cad.query_start_loc,
                num_rejected_tokens_ptr=(
                    num_rejected_tokens_gpu if has_num_rejected else 0
                ),
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=self._draft_block_size_by_gid.get(
                    kv_cache_gid, self.block_size
                ),
                num_query_per_req=num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=num_context,
                BLOCK_SIZE=block_size_tokens,
                HAS_NUM_REJECTED=has_num_rejected,
            )

        primary_kv_cache_gid = draft_kv_group_ids[0]
        query_slot_mapping = self._slot_mapping_buffers_by_gid[primary_kv_cache_gid][1][
            :num_query_total
        ]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=self._get_dflash_block_table(primary_kv_cache_gid, cad),
            slot_mapping=query_slot_mapping,
            causal=self.dflash_causal,
        )

        return num_query_total, token_indices_to_sample, new_cad

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        del mm_embed_inputs
        num_context = self._dflash_num_context
        context_commit_mask = (
            self._context_commit_mask_buffer[:num_context]
            if self._committed_prefix_kv
            else None
        )

        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,
            self._context_positions_buffer[:num_context],
            self._get_dflash_context_slot_mapping(num_context),
            context_commit_mask=context_commit_mask,
        )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del spec_step_idx
        assert self._dspark_anchor_token_ids is not None
        self._last_confidence_logits = None
        self._last_scheduled_draft_lengths = None
        if not self._confidence_schedule:
            return self.model.sample_dspark_tokens(
                hidden_states,
                self._dspark_anchor_token_ids,
                sampling_metadata,
                enable_probabilistic=(
                    self._enable_probabilistic_draft_probs
                    and not sampling_metadata.all_greedy
                ),
                use_fp64_gumbel=self.use_fp64_gumbel,
            )

        draft_token_ids, draft_probs, confidence_logits = self.model.sample_dspark_tokens(
            hidden_states,
            self._dspark_anchor_token_ids,
            sampling_metadata,
            enable_probabilistic=(
                self._enable_probabilistic_draft_probs
                and not sampling_metadata.all_greedy
            ),
            use_fp64_gumbel=self.use_fp64_gumbel,
            return_confidence=True,
        )
        self._last_confidence_logits = confidence_logits
        self._last_scheduled_draft_lengths = self._schedule_draft_lengths(
            confidence_logits
        )
        return draft_token_ids, draft_probs
