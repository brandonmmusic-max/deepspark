# Benchmark — local b12x image (3-run average)

Single-user decode + prefill on the local DeepSeek-V4-Flash-DSpark b12x stack,
averaged over **3 full runs** of the standard `llm_decode_bench`.

## Configuration
- **Hardware:** 4× RTX PRO 6000 Blackwell (96 GB, SM120), PCIe Gen5, no NVLink, **300 W/GPU**
- **Image:** `voipmonitor/vllm:chthonic-…-b12x0ff2847-pr20-cu132` (b12x runtime, our dense-GEMM port live — see [B12X_CHANGES.md](B12X_CHANGES.md))
- **Parallelism:** TP=4 · **Draft:** DSpark γ=5 (`num_speculative_tokens=5`, `draft_sample_method=probabilistic`)
- **KV:** fp8 · **Attention:** `B12X_MLA_SPARSE` · `GRAPH_CAP=64` · `gpu_mem_util=0.89`
- **Sampling:** temp **0.1** (production temp) · concurrency 1 · `ignore_eos` sustained decode

## Decode (tok/s, concurrency 1)

| context | run 1 | run 2 | run 3 | **mean** |
|---|---:|---:|---:|---:|
| 0 (ctx0) | 235.1 | 234.4 | 234.7 | **234.7** |
| 1k | 294.9 | 317.3 | 365.4 | **325.9** |
| 10k | 307.5 | 301.1 | 345.1 | **317.9** |
| 100k | 291.3 | 291.5 | 313.4 | **298.7** |
| 128k | 261.7 | 272.6 | 242.4 | **258.9** |

> **Read the variance, not just the mean.** Only **ctx0 (234.7) is rock-stable**
> (±0.4 across runs). Every context with synthetic padding is high-variance
> (run-to-run swing of ±20–35 tok/s) because DSpark draft acceptance on the
> benchmark's padding is unstable even at temp 0.1. Treat ctx0 as the reliable
> single-user number and the padded-context means as approximate (they would need
> ~10 runs to tighten). 128k in particular ranges 242–294 across runs.

## Prefill (tok/s, reference — excluded from decode)

| context | mean prefill tok/s |
|---|---:|
| 8k | 7,267 |
| 10k | 7,716 |
| 64k | 6,996 |
| 100k | 6,388 |
| 128k | 6,013 |

Prefill is a one-time per-request cost and is **not** part of the decode numbers above.

## Method
- `llm_decode_bench.py --port <…> --concurrency 1 --contexts 0,1k,10k,100k,128k --temperature 0.1`, run 3×; means computed across the 3 runs.
- Lossless: greedy output verified byte-identical separately (Estonia 30/30 on the same stack).
