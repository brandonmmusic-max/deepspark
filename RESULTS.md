# Results — `llm_decode_bench` (DeepSeek-V4-Flash + DSpark, b12x, SM120)

All numbers are the **standard `llm_decode_bench`**: sustained single-user decode
throughput, `ignore_eos`, streamed OpenAI-API usage timing, **lossless** (greedy
parity verified separately). Server: `dsv4-dspark-test` on `:9406`, DSpark
`num_speculative_tokens=5`, `draft_sample_method=probabilistic`.

The metric is decode tok/s **only** — prefill is timed and reported separately
and is **excluded** from these decode numbers.

---

## 1. Production-temperature baseline (the headline)

The real production sampling temperature is **0.0–0.1** (not vLLM's default 1.0).
Clean stack, three repeats each:

| Temp | Context | Mean tok/s | Repeats |
|---:|---|---:|---|
| 0.0 | ctx0 | **234.01** | 238.76, 227.31, 235.95 |
| 0.0 | 128k | 257.69 | 250.51, 231.03, 291.53 |
| 0.1 | ctx0 | **235.51** | 235.36, 233.54, 237.62 |
| 0.1 | 128k | 283.46 | 297.48, 272.22, 280.68 |

Observations:
- **ctx0 is rock-stable** (±~2 tok/s across repeats).
- **128k is high-variance** (spread of 25–60 tok/s). See [CAVEATS.md](CAVEATS.md).
- ctx0 < 128k. This is a real spec-decode-acceptance effect, **not** a timer bug —
  full explanation in [CAVEATS.md](CAVEATS.md#why-ctx0-is-slower-than-128k).

The widely-cited **~225 t/s** baseline was measured at vLLM's **default
`temperature=1.0`** (the sustained harness previously omitted an explicit
temperature). At production temp the stable ctx0 figure is **~235 t/s**.

---

## 2. Optimization-lever campaign — toward 300 t/s ctx0

Goal: reach **300 t/s at ctx0** on this standard bench via real kernel/code work,
lossless. Every promoted change had to beat sustained ctx0 **and** not regress
128k (the 128k canary). Outcome by lever:

| Lever | Bucket | Result | Notes |
|---|---|---|---|
| Dense split-K | Dense GEMM | **WALL** | Failed lossless parity oracle even after divisor-only + FP32-reduce fixes |
| Committed-prefix KV | MLA | **WALL** | Passed greedy correctness, regressed sustained bench (orig + compact-once) |
| MXFP8 swap-AB (large-N) | Dense GEMM | **WALL** | Byte-exact but **slower** than the unswapped path (micro + sustained) |
| External FP4 (cuDNN/CUTLASS) | Dense GEMM | **WALL** | b12x already beats external FP4 on the tested decode shapes |
| `MAX_ACTIVE_CLUSTERS` | MoE | **WALL** | 84→ctx0 up/128k cratered; 107→tiny ctx0/128k regress; 127→128k up/ctx0 regress |
| NVFP4 m==1 FC1 retile | MoE | **WALL (perf)** | Lossless + Estonia 30/30, but ctx0 −2.5% / 128k +6.7% → fails the ctx0 gate |
| **Share-input-across-experts (m>1)** | MoE | **PROMOTED** | Estonia 30/30; nsys launch calls 716,276→706,484 at equal tokens; too small to move the ceiling |

**Conservative promoted canary: 225.47 / 244.32 tok/s (ctx0 / 128k).**

### The honest wall

300 t/s at ctx0 was **not reached.** The cheap measurement levers are consumed;
the only promoted keeper (MoE share-input) is real but too small to change the
ceiling. The decode-time budget (NCU buckets) is roughly: dense GEMM ~33.7%,
PCIe one-shot all-reduce ~17.3%, MoE micro ~12.8%, MLA ~5.5%. AR and MLA are
near-tapped (hidden=4096 keeps the all-reduce payload under the 64 KB one-shot
cap). Reaching 300 ctx0 needs a **non-trivial new kernel path** (grouped/
persistent dense decode GEMM, fused dequant/projection, or a final-head
replacement) **or** higher **draft acceptance** (a larger / tree draft head) —
the latter is the only multiplicative lever left and is the subject of ongoing
DSpark-tree design work.

### Note on the NVFP4 FC1 retile (toggleable)

The NVFP4 m==1 FC1 4-rows/warp retile is **lossless** (greedy Estonia 30/30) and
is a real **long-context** win (**128k +6.7% → ~302 tok/s**) at a **ctx0** cost
(**−2.5%**). It is not promoted as default because the campaign metric is ctx0,
but for a long-context-dominated workload it is a legitimate, gated option.

---

## 3. Prefix caching (end-to-end latency, not decode rate)

Prefix caching is **enabled and working** alongside DSpark + `B12X_MLA_SPARSE` +
the sparse indexer + fp8 KV. It does **not** change decode tok/s — it skips
repeated prefill KV computation.

| Test | Cold | Hot (cached prefix) | Speedup |
|---|---:|---:|---:|
| Prefill (12.7k shared-prefix tokens) | 1.517 s | 0.242 s | **6.26×** |
| TTFT | 1.540 s | 0.268 s | **5.75×** |

- Hot request hit **12,544** cached prefix tokens, computed only **167**.
- Lossless: cold and hot same-prompt output identical (both 4829 tokens).
- **Caveat:** DCP > 1 (decode context parallel) with the DeepSeek-V4 hybrid /
  DFlash-replicated draft groups takes a code path that **disables** prefix-cache
  hits. Keep DCP = 1 to retain the cache.

A real win for repeated long-document / shared-KG-context workloads (lower TTFT
and E2E latency), distinct from the decode-throughput numbers above.

---

## 4. Prefill throughput (reference, excluded from decode numbers)

| Context | Prefill tok/s |
|---|---:|
| 8k | ~7,731 |
| 16k | ~7,054 |
| 32k | ~7,143 |
| 64k | ~7,121 |
| 128k | ~6,241 |

Prefill is a one-time cost per request and is **not** part of the decode tok/s in
Sections 1–2.

---

## Raw artifacts

Unedited campaign result docs are in [results/](results/):
- `raw_campaign_result.md` — the full 300-tps lever campaign log
- `prefix_cache_result.md` — the prefix-cache investigation
- `llm_decode_bench_full_grid_temp1.0.txt` — the original temp-1.0 grid
