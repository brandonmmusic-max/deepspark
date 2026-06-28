# DSpark 300 t/s Campaign Result

Last updated: 2026-06-28 EDT / 2026-06-28 UTC

## Metric Correction - Binding

Only the standard sustained `llm_decode_bench` single-user decode metric counts
for this campaign: `ignore_eos`, concurrency 1, byte-identical/lossless output,
as measured by the benchmark. The rejected real-legal thinking-on harness is not
a deliverable metric here because it double-counts thinking+output tokens.

Target: standard bench `ctx0 >= 300 t/s` and `16k >= 300 t/s`.

Current status after R1/R2 inspection, the R3 dense-GEMM lever pass, and the R4
MoE bucket pass:

| Run | ctx0 standard t/s | Delta vs 225.5 | 16k standard t/s | Delta vs 278.3 | Verdict |
|---|---:|---:|---:|---:|---|
| Honest baseline from directive | 225.5 | 0.0 | 278.3 | 0.0 | baseline |
| R1 sync-removal gate | 227.1 | +1.6 | 271.5 | -6.8 | fails target |
| Current final standard run | 224.8 | -0.7 | 327.8 | +49.5 | 16k passes, ctx0 fails |

Hard status: not solved. The remaining failure is ctx0 standard decode, still
~75 t/s short of 300.

The current dense canary run from R3, using the standard bench at `ctx0,128k`,
was `227.7 t/s` at ctx0 and `249.6 t/s` at 128k with the dense swap lever
disabled. Dense-only work did not move ctx0 into the 300 t/s range.

The current MoE canary from R4 promoted one small lossless code change. Clean
no-button `ctx0,128k` baseline was `222.25 / 234.15 t/s`; the promoted repeat
run was `225.47 / 244.32 t/s`. This is a real MoE win, but not a route to 300
t/s by itself.

Scope guardrails held:
- Used isolated `dsv4-dspark-test` on `:9406` only.
- Did not touch production `:9200`, `dsv4-9200-prod`, the official base model directory, or `zz_serve_dsv4_v4_chthonic_9200.sh`.
- Did not use `nvidia-smi -pl`.
- Launcher-only knobs remain default-off through environment variables in
  `zz_serve_dsv4_dspark_9406.sh`. R4 Lever 3 is an active b12x overlay code
  promotion.

## R0 Decision - Superseded

The earlier R0 decision/reframe is superseded by the metric correction above.
Real-legal thinking-on measurements remain historical diagnostics only; they do
not count as progress against the 300 t/s deliverable.

## R0a Clock-Lock Test

Supported maximum clocks from `nvidia-smi -q -d SUPPORTED_CLOCKS`:
- graphics: 3090 MHz
- memory: 14001 MHz

Applied, with `sudo -n` because direct user lock lacked permission:
- `nvidia-smi -lgc 3090`
- `nvidia-smi -lmc 14001`

Reset after the test:
- `nvidia-smi -rgc`
- `nvidia-smi -rmc`

Post-reset check shows idle clocks back at graphics 180 MHz and memory 405 MHz
on all four GPUs.

Sustained ignore_eos C=1, same ctx0/16k/32k bench:

| Mode | Context | Aggregate t/s | Server gen t/s | Spec accept rate | GPU temp max | Power avg |
|---|---:|---:|---:|---:|---:|---:|
| unlocked | 0 | 218.8 | 218.6 | 0.314 | 69 C | 1034 W |
| unlocked | 16k | 311.3 | 311.2 | 0.351 | 77 C | 1067 W |
| unlocked | 32k | 292.7 | 292.5 | 0.325 | 83 C | 1078 W |
| locked | 0 | 224.9 | 224.8 | 0.288 | 88 C | 1107 W |
| locked | 16k | 340.5 | 340.4 | 0.630 | 89 C | 1092 W |
| locked | 32k | 276.9 | 276.7 | 0.588 | 89 C | 1111 W |

Clock-lock attribution:
- ctx0: +2.8%
- 16k: +9.4%
- 32k: -5.4%

Conclusion: not a keeper as-is. It helps one sustained context but hurts 32k and
runs hot. Clocks were reset.

Artifacts:
- `dspark_300tps_artifacts/r0_clock_lock/supported_clocks.txt`
- `dspark_300tps_artifacts/r0_clock_lock/unlocked_c1_ctx0_16k_32k.json`
- `dspark_300tps_artifacts/r0_clock_lock/locked_c1_ctx0_16k_32k.json`

## R0b Real Legal Thinking-On Baseline

Harness:
- `dspark_real_legal_bench.py`
- endpoint `http://127.0.0.1:9406`
- served model `deepseek-v4-flash-dspark-test`
- thinking on via chat template kwargs, `reasoning_effort=max`
- low-temp legal prose prompts, exact prompt sizing through `/tokenize`
- `max_tokens=2048`

| Context target | Temp | Prompt tokens | Decode t/s | E2E t/s | Mean accept len | Draft accept rate |
|---:|---:|---:|---:|---:|---:|---:|
| 16k | 0.2 | 16,392 | 358.5 | 262.3 | 4.45 | 0.690 |
| 32k | 0.2 | 32,754 | 425.0 | 261.5 | 5.37 | 0.874 |
| 64k | 0.2 | 65,554 | 404.8 | 194.6 | 5.30 | 0.859 |
| 128k | 0.2 | 131,086 | 391.0 | 116.1 | 5.43 | 0.886 |
| 150k | 0.2 | 153,613 | 216.8 | 132.5 | 3.06 | 0.413 |
| 150k | 0.0 | 153,613 | 384.9 | 342.5 | 5.43 | 0.886 |
| 150k | 0.3 | 153,613 | 276.6 | 253.3 | 3.95 | 0.589 |

Conclusion: 150k is not uniformly worse than ctx0; at temp 0.0 it clears 300.
The failure mode is temperature-sensitive acceptance collapse at 150k.

Artifacts:
- `dspark_300tps_artifacts/r0_real_legal/legal_grid_temp02.json`
- `dspark_300tps_artifacts/r0_real_legal/legal_150k_temp00_03.json`
- per-context interval sample JSONs under `dspark_300tps_artifacts/r0_real_legal/`

## R0c nsys Profile

Host-side system-wide nsys run produced a report but no CUDA kernel data:
- `dspark_300tps_artifacts/r0_nsys/dspark_live_node.nsys-rep`

In-container nsys was added behind default-off env vars:
- `NSYS_PROFILE=1`
- `NSYS_OUTPUT=/cache/nsys/...`
- `NSYS_DURATION=...`
- `--cuda-graph-trace=node`

The first in-container attempt drove a live 16k legal temp 0.2 request at 408.3
decode t/s with mean accepted length 5.14, but nsys killed the server at the
duration boundary and hung before exporting the `.nsys-rep`. The wrapper now
uses `--wait=primary` for the next attempt.

Useful live log evidence captured during that attempt:
- DSpark JIT kernels appeared: `_prepare_dspark_inputs_kernel`, `_rejection_kernel`, `_resample_kernel`, `_insert_resampled_kernel`.
- vLLM spec metrics for the profiled request: mean acceptance length 5.14,
  accepted 828, drafted 1000, average draft acceptance 82.8%.

Conclusion: profiling setup was partially validated but CUDA-kernel report
capture is still incomplete.

## R0d CPU / NUMA / IRQ

Findings:
- CPU scaling governor was already `performance`.
- `cpupower` was not installed in the shell path (`sudo: cpupower: command not found`).
- `numactl -H` showed a single NUMA node.
- `nvidia-smi topo -m` showed all GPUs with CPU affinity `0-47`, NUMA node 0.
- NVIDIA IRQ affinity was already spread across CPUs `0-47`.

Conclusion: no NUMA or IRQ pin was applied. This is effectively a no-op lever on
the current single-NUMA host layout.

## R0e Gamma Sweep

Sustained ignore_eos C=1, ctx0/16k/32k:

| Gamma | Context | Aggregate t/s | Server gen t/s | Spec accept rate | GPU temp max | Power avg |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 228.9 | 228.8 | 0.465 | 78 C | 1043 W |
| 3 | 16k | 268.9 | 268.8 | 0.620 | 83 C | 1071 W |
| 3 | 32k | 247.8 | 247.7 | 0.504 | 87 C | 1079 W |
| 4 | 0 | 224.1 | 224.1 | 0.435 | 89 C | 1101 W |
| 4 | 16k | 306.2 | 306.1 | 0.801 | 89 C | 1095 W |
| 4 | 32k | 262.9 | 262.8 | 0.439 | 89 C | 1109 W |
| 5 | 0 | 219.5 | 219.4 | 0.348 | 88 C | 1103 W |
| 5 | 16k | 284.3 | 284.1 | 0.388 | 90 C | 1116 W |
| 5 | 32k | 306.1 | 305.9 | 0.861 | 89 C | 1128 W |

Gamma 6 was rejected by the live DSpark validator before serving:

`Value error, num_speculative_tokens:6 must be divisible by n_predict=5`

Gamma 7 was not launched after gamma 6 failed because it is the same
non-divisible class for the current `n_predict=5` validator.

Conclusion: no non-default gamma was promoted, so no Estonia 30/30 keeper gate
was run for gamma. Gamma 5 remains the default restored server state.

Artifacts:
- `dspark_300tps_artifacts/r0_gamma/gamma_3_c1_ctx0_16k_32k.json`
- `dspark_300tps_artifacts/r0_gamma/gamma_4_c1_ctx0_16k_32k.json`
- `dspark_300tps_artifacts/r0_gamma/gamma_5_c1_ctx0_16k_32k.json`
- `dspark_300tps_artifacts/r0_gamma/gamma_6_7_invalid.md`

## R1 Code Fixes

### R1a Gamma Loop Graph Check

Finding: no code patch applied. The current DSpark speculator already captures
`_generate_draft(...)` through `query_cudagraph_manager.capture(...)`, and
`sample_dspark_draft(...)` is called inside `_generate_draft`. The live startup
logs show `Capturing dspark CUDA graphs (FULL)` completing successfully. I did
not find an active eager host-serial gamma loop outside the DSpark draft graph
in the served overlay.

### R1b Host Sync Removal

Applied code changes:
- `dspark_b12x_overlay/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`
- `dspark_p2d_kernel_work/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`
- `dspark_b12x_overlay/vllm/v1/spec_decode/dspark.py`
- `dspark_p2d_kernel_work/vllm/v1/spec_decode/dspark.py`

Changes:
- Replaced the active per-propose `seq_lens_cpu_upper_bound[:num_reqs].max().item()`
  path with cached CPU-side `input_batch.max_seq_len_upper_bound`.
- Removed unused budget helper methods that extracted tensor budget values via
  `.item()`.
- Mirrored the cached-bound cleanup in the p2d adaptive helper. Remaining
  `.item()` calls in that p2d-only adaptive branch are behind default-off
  `dspark_confidence_schedule=false` / `DSPARK_ENABLE_CUTE_VERIFY_LENS` paths
  and are not in the served default R1 path.

Validation:
- `python3 -m py_compile` passed for all four edited files.
- Live container source check confirmed the served speculator uses
  `max_seq_len_upper_bound`, does not use the removed `seq_lens.max().item()`
  active path, and no longer has the removed budget helper.
- Parity oracle: PASS for max sequence length M=1..8.
- Estonia greedy gate: PASS 30/30.

R1 standard sustained bench:

| Context | Baseline t/s | R1 t/s | Delta |
|---:|---:|---:|---:|
| 0 | 225.5 | 227.1 | +1.6 |
| 16k | 278.3 | 271.5 | -6.8 |

R1 verdict: the host-sync cleanup is correct and low risk, but it does not close
the standard-bench gap.

Artifacts:
- `dspark_300tps_artifacts/r1_sync_current/oracle_max_seq_len_m1_8.txt`
- `dspark_300tps_artifacts/r1_sync_current/estonia_30_greedy.json`
- `dspark_300tps_artifacts/r1_sync_current/estonia_30_greedy.stdout`
- `dspark_300tps_artifacts/r1_sync_current/standard_c1_ctx0_16k.json`
- `dspark_300tps_artifacts/r1_sync_current/standard_c1_ctx0_16k.stdout`

## R2 Kernel Inspection

### R2a Dense/Projection GEMM

The current local p2d dense kernel already includes the small-M decode policy:
`direct_one_m_tile_scheduler`, `use_m1_non_tma`, `single_work_tile_per_cta`, and
2-way split-K for eligible FP8/BF16 M<=8/N>=4096/K>=4096 decode shapes. The live
image has this same dense policy installed.

I attempted current DSpark NCU precision attribution under isolated `:9406`:
- broad kernel pass: invalid for timing; NCU ended with `LaunchFailed`.
- dense-only pass: invalid for timing; NCU ended with `LaunchFailed`.
- dense-only captured kernel identity was
  `b12xgemmdenseDenseGemmKernel...Valuetypef8E4M3FN...`, so the sampled dense
  row is FP8, not W4A16. I did not capture a clean W4A16 dense timing row in
  the current DSpark profile attempt.

Because the current NCU timing was invalid, I did not tune from its zero/nan
duration counters. Historical clean NCU still marks dense/projection as a real
underfilled bucket, but the simple retile/split-K attempts already recorded in
`b12x_gqa_build/DSV4_SPEEDUP_250_RESULT.md` failed live decode gates:
- r34 tiny-M 16x64 retile: ctx0 217.61 t/s; do not promote.
- r35 splitK4/auto tiny tile: ctx0 222.88 t/s and 131k 211.05 t/s; do not promote.

R2a verdict: no safe dense code promotion was available in this workspace. The
remaining dense path needs a real grouped/persistent dense decode GEMM or fused
dequant/projection implementation, not another env or simple retile.

Artifacts:
- `dspark_300tps_artifacts/r2a_ncu_precision_current/run_dspark_ncu_precision.py`
- `dspark_300tps_artifacts/r2a_ncu_precision_current/server_full.log`
- `dspark_300tps_artifacts/r2a_ncu_precision_current/profile_raw.csv`
- `dspark_300tps_artifacts/r2a_ncu_precision_current/dense_gemm/server_full.log`

## Rank 1 Wave-Aware Split-K

Applied code changes:
- `/home/brandonmusic/KLC_SANDBOXES/b12x/b12x/gemm/dense.py`
  - Added default-off `B12X_DENSE_SPLITK_WAVE`.
  - Added divisor-guarded wave split-K chooser.
  - Generalized the non-atomic split-K reducer from 2-way to N-way FP32.
  - Kept `B12X_DENSE_SPLITK_TURBO` separate; the rank-1 oracle was run with
    turbo off.
- `/home/brandonmusic/KLC_SANDBOXES/b12x/tests/test_gemm_stack.py`
  - Added default-off policy coverage, narrow-N wave policy coverage, known
    decode-shape chooser coverage, and non-divisible K-tile fallback coverage.
- `/home/brandonmusic/klc-linux/zz_serve_dsv4_dspark_9406.sh`
  - Added env pass-through for `B12X_DENSE_SPLITK_TURBO`,
    `B12X_DENSE_SPLITK_WAVE`, and `B12X_DENSE_SPLITK_WAVE_MAX` so isolated
    `:9406` runs can keep turbo off for lossless probes.

Validation:
- `python3 -m py_compile` passed for `dense.py` and `test_gemm_stack.py`.
- `pytest -q tests/test_gemm_stack.py -k 'splitk_wave_policy or splitk_wave_chooser'`
  passed: `5 passed`.
- Direct N-way Triton reducer check passed: `equal True`, `max_abs 0.0`.
- Local host CUDA MXFP8 dense compile skipped because the host CUTLASS DSL lacks
  `MmaMXF8Op`; the live-image disposable container has `MmaMXF8Op`.
- Live-image disposable dense oracle, with patched b12x mounted and
  `B12X_DENSE_SPLITK_TURBO=0`, confirmed policy:
  - `N=2816,K=7168 -> split_k=4`
  - `N=2816,K=7552 -> split_k=1` because 59 K-tiles has no divisor in
    `{2,4,6,8}`
  - `N=4096,K=7168 -> split_k=2`
- Live-image disposable dense oracle failed exact parity for the new narrow-N
  split: `M=1,N=2816,K=7168`, split-4 vs unsplit produced
  `parity_equal False`, `parity_max_abs 7.62939453125e-06`.

Standard bench:
- Not run for rank 1. The lossless oracle failed before the standard-bench gate,
  so benchmarking it would not be a keeper result.

Keeper verdict: fail closed. Do not promote rank 1. Move to rank 2.
- `dspark_300tps_artifacts/r2a_ncu_precision_current/dense_gemm/profile_raw.csv`

## Rank 2 Committed-Prefix Draft KV

Applied code changes:
- `/home/brandonmusic/klc-linux/dspark_b12x_overlay/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`
  - Added default-off `DSPARK_COMMITTED_PREFIX_KV`.
  - Added a per-context-row `context_commit_mask` derived from
    `valid_ctx_end = ctx_end - num_rejected`.
  - Passed the mask to `precompute_and_store_context_kv(...)` only when the env
    is enabled.
- `/home/brandonmusic/klc-linux/dspark_b12x_overlay/vllm/v1/spec_decode/dspark.py`
  and `/home/brandonmusic/klc-linux/dspark_b12x_overlay/vllm/v1/spec_decode/utils.py`
  - Mirrored the same default-off mask path for the V1 proposer route, though the
    live served container used the worker `gpu/spec_decode/dspark/speculator.py`
    route.
- `/home/brandonmusic/klc-linux/dspark_b12x_overlay/vllm/models/deepseek_v4/nvidia/dspark.py`
  - Added optional `context_commit_mask` support and compacted
    `context_states`, `context_positions`, and slot mappings before
    `fused_wqa_wkv(...)`.
- `/home/brandonmusic/klc-linux/zz_serve_dsv4_dspark_9406.sh`
  - Added `DSPARK_COMMITTED_PREFIX_KV` pass-through, default `0`.

Validation:
- `python3 -m py_compile` passed for the active worker speculator, V1 proposer
  mirror, DSpark utils, and DeepSeek DSpark model file.
- Live served source check confirmed the mounted container saw
  `context_commit_mask` in the worker speculator, V1 proposer, and model
  signature.
- Live startup logs on `:9406` confirmed all four workers loaded the active
  route with `DSpark committed-prefix context KV precompute is enabled.`
- Estonia greedy gate passed: `30/30`, `PASS 30 / FAIL 0`.

Rank 2 standard sustained bench:

| Context | Current t/s | Rank 2 t/s | Delta | Spec accept rate | Verdict |
|---:|---:|---:|---:|---:|---|
| 0 | 224.8 | 204.8 | -19.9 | 0.337 | FAIL |
| 16k | 327.8 | 263.3 | -64.5 | 0.381 | FAIL |

Benchmark source: `openai_continuous_usage` for both cells.

Artifacts:
- `dspark_300tps_artifacts/r2_committed_prefix_kv/estonia_30_greedy.json`
- `dspark_300tps_artifacts/r2_committed_prefix_kv/estonia_30_greedy.stdout`
- `dspark_300tps_artifacts/r2_committed_prefix_kv/standard_c1_ctx0_16k.json`
- `dspark_300tps_artifacts/r2_committed_prefix_kv/standard_c1_ctx0_16k.stdout`

Keeper verdict: fail closed. Do not promote rank 2. The standard metric
regressed at both required contexts, including a 16k drop below the already
passing 327.8 t/s gate.

### R2b mHC Retarget

The requested retarget is already present in the active overlay path. With
`VLLM_USE_B12X_MHC=1`, `DeepseekV4Attention._run_b12x_mhc_post_pre(...)` imports
and calls `b12x.integration.residual.b12x_mhc_post_pre`; the launcher sets
`VLLM_USE_B12X_MHC=1` and `B12X_MHC_MAX_TOKENS=16384`.

The remaining TileLang calls are final-head paths (`hc_head_fused_kernel_tilelang`)
in model/mtp/dspark code, not the dead `mhc_fused_post_pre` target called out in
the plan. I did not replace those without a verified b12x final-head equivalent.

R2b verdict: no retarget patch to apply; the intended `post_pre` path is already
the active b12x path.

### R2 Current Standard Bench

After restoring the normal isolated DSpark server on `:9406`, I reran the
standard sustained bench:

| Context | Baseline t/s | Current t/s | Delta | Target status |
|---:|---:|---:|---:|---|
| 0 | 225.5 | 224.8 | -0.7 | FAIL |
| 16k | 278.3 | 327.8 | +49.5 | PASS |

The benchmark source is `openai_continuous_usage` for both cells. The 16k cell
now passes the 300 target, but the directive requires ctx0 and 16k, so the
campaign is still failing on ctx0.

Artifacts:
- `dspark_300tps_artifacts/r2_current_standard/standard_c1_ctx0_16k.json`
- `dspark_300tps_artifacts/r2_current_standard/standard_c1_ctx0_16k.stdout`

## Rank 3 External Same-Silicon FP4 Oracle

Built/prepared `/home/brandonmusic/flashinfer-pr` for local JIT use:
- Ran the repo editable prep so `flashinfer/data/{csrc,include,cutlass,spdlog}`
  exists for JIT builds.
- Confirmed upstream PR #3152 is merged and provides SM120/121 small-N
  blockscaled GEMM coverage. The local checkout is Brandon's
  `sm120-k64-blockscaled-moe-gemm` branch, not pristine upstream #3152; it has
  six local K=64/sm120 branch commits and a modified CUTLASS submodule.
- Host FlashInfer cuDNN/CUTLASS graph oracle succeeded, but the clean b12x
  comparison required the serving image because host CUTLASS DSL lacked
  `partition_fragment_SFA`/`MmaMXF8Op`.

Final oracle was run inside the isolated serving image without starting a server
or binding ports:
- image: `voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132`
- GPU: single RTX PRO 6000 Blackwell SM120
- env: `B12X_DENSE_SPLITK_WAVE=0`, `B12X_DENSE_SPLITK_TURBO=0`
- timing: CUDA graph replay, shared b12x-quantized NVFP4 operands
- correctness: b12x vs FlashInfer CUTLASS and cuDNN was `max_abs=0.0` for every
  row.

Rank 3 FP4 decode-shape oracle:

| M | N | K | b12x us | FI CUTLASS us | FI cuDNN us | b12x/cuDNN |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2112 | 7168 | 24.6 | 32.8 | 26.6 | 0.92 |
| 1 | 2816 | 7168 | 26.6 | 32.8 | 28.7 | 0.93 |
| 1 | 4096 | 7168 | 28.7 | 34.8 | 32.8 | 0.88 |
| 1 | 5376 | 7168 | 28.7 | 41.0 | 36.9 | 0.78 |
| 1 | 7168 | 7168 | 30.7 | 45.1 | 45.1 | 0.68 |
| 6 | 2112 | 7168 | 24.6 | 32.8 | 26.6 | 0.92 |
| 6 | 2816 | 7168 | 28.7 | 32.8 | 28.7 | 1.00 |
| 6 | 4096 | 7168 | 28.7 | 36.9 | 30.7 | 0.93 |
| 6 | 5376 | 7168 | 30.7 | 41.0 | 34.8 | 0.88 |
| 6 | 7168 | 7168 | 32.8 | 45.1 | 38.9 | 0.84 |

Summary:
- cuDNN beats FlashInfer CUTLASS on 9/10 rows and ties one row, so the external
  oracle confirms cuDNN is the stronger FlashInfer FP4 backend on this SM120
  setup.
- b12x beats cuDNN on 9/10 rows and ties one row. Geomean b12x/cuDNN is
  `0.872`; geomean b12x/CUTLASS is `0.763`.
- Therefore Rank 3 does not expose a near-drop-in external FP4 dense win. It
  bounds the drop-in FP4 headroom as already consumed by b12x for these decode
  shapes.

Artifacts:
- `dspark_300tps_artifacts/r3_flashinfer_oracle/flashinfer_mm_fp4_oracle.py`
- `dspark_300tps_artifacts/r3_flashinfer_oracle/b12x_flashinfer_fp4_decode_oracle.py`
- `dspark_300tps_artifacts/r3_flashinfer_oracle/mm_fp4_oracle_custom_graph.csv`
- `dspark_300tps_artifacts/r3_flashinfer_oracle/b12x_vs_flashinfer_fp4_decode_container_all.csv`
- `dspark_300tps_artifacts/r3_flashinfer_oracle/b12x_vs_flashinfer_fp4_decode_container_all.json`

## R3 Dense-GEMM Methodology Fix

Directive scope: iterate the three dense levers to `PROMOTED` or `WALL` on the
isolated `dsv4-dspark-test` stack at `:9406`. Production `:9200`,
`dsv4-9200-prod`, the official base model, clocks, and power limits were not
touched.

Code changes applied:
- `dspark_p2d_kernel_work/b12x/gemm/dense.py`
  - Added fail-closed `B12X_DENSE_SPLITK`, `B12X_DENSE_SPLITK_LOSSLESS`,
    `B12X_DENSE_SPLITK_MAX`, `B12X_DENSE_MXFP8_SWAP_AB`, and
    `B12X_DENSE_MXFP8_SWAP_TILE_N`.
  - Added divisor-only split-K selection and an ordered FP32 split-K reducer.
  - Routed large-N MXFP8 decode through `(64,32)` or `(64,16)` swap-AB when
    explicitly enabled; the `(64,32)/(64,16)` stage cap was already present.
- `benchmarks/probe_dense_fp8_tile_sweep.py`
  - Added byte-identical MXFP8 large-N swap parity and preallocated timing sweep
    for `(16,128)` unswapped, `(64,32)` swap, and `(64,16)` swap.
- `zz_serve_dsv4_dspark_9406.sh`
  - Added pass-through for the new dense env controls so `:9406` runs can
    toggle the patched b12x overlay without touching prod.
- `dspark_b12x_overlay/vllm/models/deepseek_v4/nvidia/dspark.py`
  - Retried committed-prefix KV with a compact-once variant: compute committed
    indices once, compact `context_states` and `context_positions` once, and
    reuse the indices for per-layer slot mappings.

### Lever A - Split-K Bit-Exactness

Research:
- NVIDIA CUTLASS documents split-K as partitioned-K GEMM plus a batched
  reduction workspace: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- CUTLASS Stream-K examples distinguish deterministic reduction order from
  non-deterministic reduction order:
  https://github.com/NVIDIA/cutlass/blob/main/examples/74_blackwell_gemm_streamk/blackwell_gemm_streamk.cu
- Colfax's CUTLASS Stream-K tutorial calls out the workspace/reduction overhead
  tradeoff in split/stream-K style decompositions:
  https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/

Root cause and iteration:
- Fixed the two concrete losslessness hazards called out in the directive:
  divisor-only split selection prevents K-tail tile dropping, and the default
  lossless path uses FP32 ordered reduction instead of bf16 atomic reduction.
- Full MXFP8 parity grid:
  - `(64,32)` swap parity: exact `20/20`.
  - `(64,16)` swap parity: exact `20/20`.
  - split-K max-2 parity: exact `13/20`, failed `7/20`.
- Split-K failures after the fixes were tiny but byte-visible BF16 differences,
  for example `M=4,N=5376,K=7168` had 3 mismatches with max abs `0.0625`.
- Legal split-factor sweep on `M=8,N=7168,K=7168` failed for every divisor
  tested: `2,4,7,8,14,28,56`. Even one K-tile per split (`56`) was not
  byte-identical.

Verdict: `WALL`. The blocker is no longer the tail floor-div bug or bf16
atomic reduction; it is the changed accumulation/reduction order of split-K on
this MXFP8 BF16-output path. It cannot enter the sustained keeper gate because
the directive requires `atol=0` parity first.

Artifacts:
- `dspark_300tps_artifacts/r3_dense_gemm/parity_full.log`
- `dspark_300tps_artifacts/r3_dense_gemm/parity_splitk56_m4_m8_n4096.log`
- `dspark_300tps_artifacts/r3_dense_gemm/parity_splitk_factor_sweep_m8_n7168.log`

### Lever B - MXFP8 Swap-AB

Parity and micro sweep:
- Byte-identical parity passed for both large-N swap candidates:
  - `(64,32)` swap: exact `20/20`.
  - `(64,16)` swap: exact `20/20`.
- Preallocated tile sweep over 20 decode shapes:
  - `(16,128)` unswapped won `20/20` by p50.
  - `(64,32)` swap was slower than unswapped with median p50 ratio `1.656x`,
    min `1.133x`, max `2.518x`.
  - `(64,16)` swap was worse with median p50 ratio `2.375x`.

Keeper gates:
- Live JIT confirmed `(64,32)` swap-AB was actually compiled on `:9406`
  (`tile=[64,32]`, `swap_ab=true`, `split_k_slices=1`).
- Estonia greedy gate passed `30/30`.
- Standard `llm_decode_bench`, `C=1`, contexts `0,128k`:

| Run | ctx0 t/s | Delta vs no-swap | 128k t/s | Delta vs no-swap | Verdict |
|---|---:|---:|---:|---:|---|
| no-swap `(16,128)` | 227.7 | 0.0 | 249.6 | 0.0 | reference |
| swap `(64,32)` | 203.0 | -24.7 | 214.1 | -35.5 | WALL |

Verdict: `WALL`. Swap-AB is correct but slower at the kernel-sweep level and
slower on the standard sustained bench at both ctx0 and 128k. It also fails the
protected ~229 canary.

Artifacts:
- `dspark_300tps_artifacts/r3_dense_gemm/tile_sweep_all_prealloc.log`
- `dspark_300tps_artifacts/r3_dense_gemm/swap32_estonia_30_greedy.json`
- `dspark_300tps_artifacts/r3_dense_gemm/swap32_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r3_dense_gemm/noswap_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r3_dense_gemm/swap32_gemm_dense_jit.log`
- `dspark_300tps_artifacts/r3_dense_gemm/noswap_gemm_dense_jit.log`

### Lever C - Committed-Prefix KV

Root cause:
- The original committed-prefix implementation passed `context_commit_mask` into
  `precompute_and_store_context_kv(...)`.
- Each DSpark layer then used PyTorch boolean indexing on `context_states`,
  `context_positions`, and its slot mapping. This adds dynamic gather/allocation
  work and extra launches to remove only the rejected speculative tail.
- The DSpark head has three target layers (`dspark_target_layer_ids [40,41,42]`),
  so the old implementation repeated at least the state and position compaction
  work across layers. The p2d variant in this workspace had already removed the
  committed-prefix path and kept the normal full-context KV path.

Iteration:
- Retried with compact-once indexing at the model level, then reused the indices
  for layer slot mappings.
- The tuned variant loaded in the container with `DSPARK_COMMITTED_PREFIX_KV=1`
  and passed Estonia greedy `30/30`.
- Standard `llm_decode_bench`, `C=1`, contexts `0,128k`:

| Run | ctx0 t/s | Delta vs no-swap | 128k t/s | Delta vs no-swap | Verdict |
|---|---:|---:|---:|---:|---|
| no-swap / committed-prefix off | 227.7 | 0.0 | 249.6 | 0.0 | reference |
| committed-prefix compact-once | 208.1 | -19.6 | 193.3 | -56.3 | WALL |

Verdict: `WALL`. The repeated boolean-indexing root cause was reduced but not
eliminated as a sustained-regression source. The remaining gather/index-select
and dynamic compaction overhead is larger than the KV work saved by omitting the
small rejected tail in this standard C=1 decode workload.

Artifacts:
- `dspark_300tps_artifacts/r3_dense_gemm/committed_prefix_compact_once_estonia_30_greedy.json`
- `dspark_300tps_artifacts/r3_dense_gemm/committed_prefix_compact_once_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r3_dense_gemm/committed_prefix_compact_once_server.log`

## R4 MoE Bucket

Directive scope: iterate the three MoE levers to `PROMOTED` or `WALL` on the
isolated `dsv4-dspark-test` stack at `:9406`, using the standard
`llm_decode_bench` ctx0 and 128k canary. Production `:9200`,
`dsv4-9200-prod`, the official base model, clocks, and power limits were not
touched.

Clean no-button baseline, re-established with dense swap disabled and no MoE
bucket knobs:

| Context | Baseline t/s | Server gen t/s |
|---:|---:|---:|
| 0 | 222.2455 | 222.1552 |
| 128k | 234.1462 | 234.0848 |

Launcher changes applied:
- `zz_serve_dsv4_dspark_9406.sh`
  - Added default-off `B12X_MICRO_MAX_ACTIVE_CLUSTERS` pass-through so the
    live `:9406` test container can exercise the already-registered micro MoE
    kernel override.
  - Added default-off `NSYS_LAUNCH_SESSION`, `NSYS_SESSION_NAME`, and
    `NSYS_CUDA_GRAPH_TRACE` controls for decode-window nsys proofs.

### Lever 1 - `B12X_MICRO_MAX_ACTIVE_CLUSTERS`

Root cause:
- The tuned MoE micro-decode occupancy ladder exists, but the micro launch path
  calls `_get_impl_mac("micro")` without `routed_rows`, so the ladder is not
  consulted on the static micro path.
- The environment override is checked before that gate, so
  `B12X_MICRO_MAX_ACTIVE_CLUSTERS` reaches the kernel without rebuilding.

Iteration:
- Swept the prescribed values `84`, `107`, and `127` on the live `:9406`
  micro path.
- Every run used the standard sustained ctx0 and 128k canary; no result was
  promoted from micro-SM or occupancy alone.

| Run | ctx0 t/s | Delta vs clean | 128k t/s | Delta vs clean | Verdict |
|---|---:|---:|---:|---:|---|
| clean no-button | 222.2455 | 0.0000 | 234.1462 | 0.0000 | reference |
| MAC 84 | 225.2730 | +3.0275 | 206.1711 | -27.9751 | WALL |
| MAC 107 | 223.6151 | +1.3696 | 225.3484 | -8.7978 | WALL |
| MAC 127 | 213.7691 | -8.4764 | 247.4667 | +13.3206 | WALL |

Verdict: `WALL`. This is the r35 occupancy trap in live form: individual
contexts move, but no setting beats the clean baseline on both ctx0 and 128k.

Artifacts:
- `dspark_300tps_artifacts/r4_moe_bucket/no_button_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r4_moe_bucket/mac84_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r4_moe_bucket/mac107_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r4_moe_bucket/mac127_standard_c1_ctx0_128k.json`

### Lever 2 - FC1 4-Rows/Warp Retile for NVFP4 m==1

Root cause:
- `b12x/moe/fused/micro.py` already contains an m==1, gated, `n<=2048`,
  k-segments==8 FC1 retile path, but it is guarded by `w4a16_mode`.
- The served DSpark MoE shape is NVFP4, so the pure tiling/reg-hoist branch was
  unreachable for the relevant path.

Iteration:
- Extended only the pure tiling/reg-hoist condition to the exact NVFP4 m==1
  shape and verified the live config selected `rows_per_warp_fc1=4`,
  `k_segments=8`, and gated activation.
- The patched server loaded and ran the standard ctx0/128k canary.
- The patch was reverted after the canary failed.

| Run | ctx0 t/s | Delta vs clean | 128k t/s | Delta vs clean | Verdict |
|---|---:|---:|---:|---:|---|
| clean no-button | 222.2455 | 0.0000 | 234.1462 | 0.0000 | reference |
| NVFP4 m==1 rows/warp=4 | 226.9230 | +4.6775 | 232.7416 | -1.4045 | WALL |

Verdict: `WALL`. The retile helps ctx0 but regresses the 128k canary, so it
cannot be promoted as a sustained standard-bench win.

Artifacts:
- `dspark_300tps_artifacts/r4_moe_bucket/lever2_nvfp4_m1_rpw4_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r4_moe_bucket/lever2_nvfp4_m1_rpw4_standard_c1_ctx0_128k.stdout`

### Lever 3 - Share Input Across Experts for m>1

Root cause:
- The static compact MoE launch had
  `share_input_across_experts=(activation in ("relu2", "silu") and m == 1 ...)`.
- This meant pure decode benefited, but MTP m=4 and DSpark m=6 verify passes
  re-quantized the same token input once per expert route.

Promoted code change:
- `dspark_p2d_kernel_work/b12x/integration/tp_moe.py`
  - Removed only the `m == 1` clamp.
  - Kept `activation in ("relu2", "silu")`, `quant_mode == "nvfp4"`,
    `a1_gscale.numel() == 1`, and the default-on escape hatch
    `B12X_MICRO_SHARE_INPUT_ACROSS_EXPERTS`.

Lossless evidence:
- The fused micro kernel's shared-input branch skips re-quantization only when
  the next route belongs to the same input token. It reuses the same shared
  memory activation layout for that token and keeps the same scalar input scale
  guard, so it removes duplicated quant work without changing expert dot-order
  math.
- Estonia greedy gate passed: `30/30`, `PASS 30 / FAIL 0`, aggregate generation
  throughput `279.1346 t/s`.

Standard sustained keeper gate:

| Run | ctx0 t/s | Delta vs clean | 128k t/s | Delta vs clean | Verdict |
|---|---:|---:|---:|---:|---|
| clean no-button | 222.2455 | 0.0000 | 234.1462 | 0.0000 | reference |
| Lever 3 first | 229.4826 | +7.2371 | 246.0115 | +11.8653 | PASS |
| Lever 3 repeat | 225.4656 | +3.2201 | 244.3170 | +10.1709 | PASS |

Nsys launch-count gate:
- First node-trace sustained window showed lower launches per generated token,
  but more generated tokens in the fixed-duration window, so I reran with
  graph-level tracing and fixed request count.
- Fixed-token profiles both produced `4,160` metric generation tokens and
  `4,096` bench output tokens.
- Runtime launch totals decreased:
  - total CUDA runtime launch calls: `716,276 -> 706,484` (`-9,792`, `-1.37%`)
  - `cudaGraphLaunch`: `12,436 -> 12,260` (`-176`, `-1.42%`)
  - `cudaLaunchKernel`/`cuLaunchKernel`: `703,840 -> 694,224` (`-9,616`,
    `-1.37%`)
- The container-side nsys importer cannot load `libdw.so.1`, so `nsys stop`
  emitted `.qdstrm`; the host 2026.1 `QdstrmImporter` converted the streams to
  `.nsys-rep`, then SQLite export provided the counts above.

Verdict: `PROMOTED`. This is a real, sustained, lossless MoE improvement on the
standard ctx0/128k canary, with fixed-token nsys proof that total decode launch
count did not rise.

Artifacts:
- `dspark_300tps_artifacts/r4_moe_bucket/lever3_share_input_mgt1_estonia_30_greedy.json`
- `dspark_300tps_artifacts/r4_moe_bucket/lever3_share_input_mgt1_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r4_moe_bucket/lever3_share_input_mgt1_standard_c1_ctx0_128k_repeat.json`
- `dspark_300tps_artifacts/r4_moe_bucket/nsys/exact_graph_launch_count_compare.txt`
- `dspark_300tps_artifacts/r4_moe_bucket/nsys/base_exact_graph.nsys-rep`
- `dspark_300tps_artifacts/r4_moe_bucket/nsys/l3_exact_graph.nsys-rep`
- `dspark_300tps_artifacts/r4_moe_bucket/nsys/base_exact_graph`
- `dspark_300tps_artifacts/r4_moe_bucket/nsys/l3_exact_graph`

## R5 FC1 Retile Noise-Band Correction

Plan:
- Correct the premature R4 Lever 2 verdict: the single 128k delta
  `232.7416 vs 234.1462` was a `-0.6%` sample, not a noise-confirmed wall.
- Patch only the active `:9406` b12x mount,
  `dspark_p2d_kernel_work/b12x/moe/fused/micro.py`.
- Extend the existing m==1 FC1 rows/warp=4 retile from W4A16 to the
  corresponding NVFP4 micro path: `m == 1`, `n <= 2048`, gated activation,
  aligned `k_segments == 8`, and not `a8_mx_mode`.
- Keep all numeric paths unchanged. NVFP4 input quantization still writes the
  same quantized/dequantized activation words into `smem_xh`; W1/W2 microscale
  loads still use the same E4M3 or E8M0 scale decoding; FC1 dot order inside
  each output row remains the existing paired dual-dot path. The change only
  lowers `fc1_chunks` for the literal DSV4 m==1 shape so each warp computes
  four FC1 output rows while reusing the already-hoisted activation registers.
- Keep the promoted share-input-across-experts m>1 lever enabled and measure
  the stack, because the live `dsv4-dspark-test` overlay already includes that
  promotion.
- Gate: local lossless/parity probe where available, Estonia greedy `30/30`,
  then repeated standard sustained `llm_decode_bench` ctx0 and 128k, at least
  three clean and three patched runs. Promote only if ctx0 improves beyond the
  observed spread and 128k is within-noise-or-better; if 128k regresses beyond
  noise, make the retile more shape/context-conditional and remeasure.

## R6 Production-Temp FC1 Retile Remeasure

Bench script inspection:
- The sustained decode path in
  `/home/brandonmusic/llm-inference-bench/llm_decode_bench.py` previously did
  not put `temperature` into the measured request payload or the context scout
  payload. The old `~225 t/s` baseline was therefore the server/model default.
- The running vLLM `SamplingParams` default is `temperature=1.0`. I added a
  no-op-by-default `--temperature` argument to the bench so production-temp
  measurements explicitly send `temperature: 0.0` or `temperature: 0.1`.
- `ignore_eos` remained enabled. This is still the standard sustained
  `llm_decode_bench` pure output tok/s path, not the legal/thinking harness.

Live server/config state before patching:
- Endpoint: isolated `dsv4-dspark-test` on `http://127.0.0.1:9406`.
- Startup logs and container command both show DSpark
  `draft_sample_method: probabilistic`.
- Active b12x mount:
  `/home/brandonmusic/klc-linux/dspark_p2d_kernel_work/b12x`.
- I restored the live FC1 gate to clean W4A16-only before the baseline:
  `if self.w4a16_mode and m == 1 and n <= 2048`.
- Dense swap-AB was kept disabled (`B12X_DENSE_MXFP8_SWAP_AB=0`) to match the
  existing promoted share-input stack; production `:9200` was not touched.

Production-temp clean baseline, standard sustained `C=1`, ctx0/128k,
`max_tokens=2048`, `ignore_eos=true`, DSpark probabilistic:

| Temp | Context | Repeats t/s | Mean t/s | Noise band (min..max) |
|---:|---|---:|---:|---:|
| 0.0 | ctx0 | 238.7614, 227.3138, 235.9494 | 234.0082 | 227.3138..238.7614 |
| 0.0 | 128k | 250.5065, 231.0250, 291.5304 | 257.6873 | 231.0250..291.5304 |
| 0.1 | ctx0 | 235.3648, 233.5448, 237.6213 | 235.5103 | 233.5448..237.6213 |
| 0.1 | 128k | 297.4779, 272.2238, 280.6754 | 283.4590 | 272.2238..297.4779 |

Artifacts:
- `dspark_300tps_artifacts/r5_prod_temp_fc1/prodtemp_t0.0_clean_rep1_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r5_prod_temp_fc1/prodtemp_t0.0_clean_rep2_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r5_prod_temp_fc1/prodtemp_t0.0_clean_rep3_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r5_prod_temp_fc1/prodtemp_t0.1_clean_rep1_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r5_prod_temp_fc1/prodtemp_t0.1_clean_rep2_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r5_prod_temp_fc1/prodtemp_t0.1_clean_rep3_standard_c1_ctx0_128k.json`

FC1 retile execution plan:
- Reapply only the gate extension in
  `dspark_p2d_kernel_work/b12x/moe/fused/micro.py`, changing the W4A16-only
  entry condition to also admit the NVFP4 m==1 DSV4 FC1 shape:
  `not a8_mx_mode`, gated activation, aligned `k_segments == 8`, `n <= 2048`.
- This is a tiling-only retile. The NVFP4 microscale decode path remains the
  same: input activation quantization still uses the existing NVFP4 quant/dequant
  into `smem_xh`, W1/W2 scale loads and alpha handling are unchanged, and the
  per-row FC1 dot order remains the existing kernel order.
- Restart only `dsv4-dspark-test` on `:9406` after patching so the mounted b12x
  module is re-imported. Keep the already-promoted share-input-across-experts
  m>1 path enabled and measure the stack.
- Run the available lossless gates: live file-line verification, py-compile
  where applicable, Estonia greedy `30/30`, then repeated standard sustained
  `llm_decode_bench` at the gated production temp `0.1`.
- Promotion rule: promote only if ctx0 improves beyond the clean temp-0.1
  ctx0 noise band and 128k is within-noise-or-better against the clean temp-0.1
  128k band. A wall requires a regression confirmed beyond this noise band over
  repeats; one 128k sample is not a wall.

FC1 retile implementation and gates:
- Applied the planned gate extension at
  `dspark_p2d_kernel_work/b12x/moe/fused/micro.py:562-574`.
- `python3 -m py_compile` passed for both the active `micro.py` and the patched
  `/home/brandonmusic/llm-inference-bench/llm_decode_bench.py`.
- Restarted only `dsv4-dspark-test` on `:9406`; live mounted file verification
  showed the NVFP4 gate extension active.
- Estonia greedy gate passed: `30/30`, `PASS 30 / FAIL 0`, aggregate generation
  `274.2 tok/s`.

Patched temp-0.1 standard sustained gate, FC1 NVFP4 m==1 rows/warp=4 stacked
with the promoted share-input m>1 path:

| State | Context | Repeats t/s | Mean t/s | Noise band (min..max) | Delta vs clean mean |
|---|---|---:|---:|---:|---:|
| clean | ctx0 | 235.3648, 233.5448, 237.6213 | 235.5103 | 233.5448..237.6213 | 0.0000 |
| patched | ctx0 | 233.4560, 226.1560, 229.0139 | 229.5420 | 226.1560..233.4560 | -5.9683 (-2.53%) |
| clean | 128k | 297.4779, 272.2238, 280.6754 | 283.4590 | 272.2238..297.4779 | 0.0000 |
| patched | 128k | 274.4648, 310.9738, 321.7601 | 302.3995 | 274.4648..321.7601 | +18.9405 (+6.68%) |

Verdict: `WALL / DO NOT PROMOTE`.

The retile is lossless enough to pass the live greedy correctness gate and it
helps 128k under temp 0.1, but it repeatedly regresses ctx0 beyond the clean
noise band. All three patched ctx0 runs are at or below the clean min
(`233.4560 <= 233.5448`, then `226.1560`, `229.0139`). This is a real
noise-confirmed wall for the promotion gate, not a one-run 128k artifact.
Because the confirmed regression is ctx0, making the retile context-conditional
would only disable it on the very context that must improve; there is no
promotion shape left from this FC1 lever.

Post-verdict cleanup: because the retile was not promoted, I restored the
active `micro.py` gate to the clean W4A16-only condition and relaunched
`dsv4-dspark-test` on `:9406`. Live file verification after restart shows
`if self.w4a16_mode and m == 1 and n <= 2048`.

Artifacts:
- `dspark_300tps_artifacts/r5_prod_temp_fc1/fc1_nvfp4_m1_rpw4_estonia_30_greedy.json`
- `dspark_300tps_artifacts/r5_prod_temp_fc1/prodtemp_t0.1_fc1_nvfp4_m1_rpw4_rep1_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r5_prod_temp_fc1/prodtemp_t0.1_fc1_nvfp4_m1_rpw4_rep2_standard_c1_ctx0_128k.json`
- `dspark_300tps_artifacts/r5_prod_temp_fc1/prodtemp_t0.1_fc1_nvfp4_m1_rpw4_rep3_standard_c1_ctx0_128k.json`

## Hard Wall

I did not reach 300 t/s at ctx0 on the standard bench. The cheap measurement
levers are now done and were no-ops/non-keepers except R4 MoE Lever 3, which
promoted but is too small to change the honest ceiling. Rank 1/R3 split-K
failed the lossless parity oracle even after divisor-only and FP32-reduce fixes. Rank 2/R3
committed-prefix KV passed greedy correctness but regressed the standard bench
after both the original and compact-once implementations. R3 swap-AB passed
parity and greedy correctness but regressed standard ctx0 and 128k. Rank 3
found no drop-in external FP4 dense win: b12x already beats FlashInfer
cuDNN/CUTLASS on the tested decode shapes. R4 MoE max-active-clusters and the
NVFP4 m==1 retile both walled on the ctx0/128k canary. R4
share-input-across-experts promoted, but the conservative repeat result is
`225.47 / 244.32 t/s` at ctx0/128k, not a 300 t/s solution. The available
inspection found that the easy dense and mHC paths are already consumed:
- current dense has the existing small-M/split-K policy installed;
- prior simple dense retile/splitK candidates regressed or failed promotion;
- large-N MXFP8 swap-AB is byte-exact but slower than the current unswapped
  path in both micro and standard sustained benches;
- current mHC `post_pre` already uses `b12x_mhc_post_pre`;
- external FP4 cuDNN/CUTLASS does not beat b12x on the decode-shape oracle;
- MoE is only about 12.8% of decode, so a single MoE-side launch-count win
  cannot honestly be reported as a path to 300 by itself;
- no ready default-off persistent dense or AR-fused-quant switch exists in this
  workspace to enable and gate.

Honest remaining work to make ctx0 reach 300 is a non-trivial new kernel path:
grouped/persistent dense decode GEMM, fused dequant/projection, final-head
replacement if a correct b12x equivalent exists, or a trained draft/tau head
that lifts the standard bench. I did not fabricate a victory or report the
forbidden real-legal harness as the deliverable.

## Current Server State

`dsv4-dspark-test` is running on `:9406` with the promoted R4 Lever 3 b12x
overlay mounted, dense swap-AB disabled, FC1 NVFP4 retile not promoted/reverted,
and no active nsys collection. Final live verification after the R6 cleanup
confirmed DSpark `draft_sample_method: probabilistic` and the clean W4A16-only
FC1 gate in the mounted b12x source. R3/R4/R6 testing used only the isolated
`:9406` test container and disposable single-GPU oracle containers. Production
`:9200`/`dsv4-9200-prod` was not touched.
