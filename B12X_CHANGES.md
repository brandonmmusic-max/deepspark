# b12x kernel changes (local fork)

This stack runs DeepSeek-V4-Flash-DSpark on the open-source b12x runtime. One
local kernel change is **live**. A second was attempted and **reverted** — it's
listed honestly so the record is complete, not because it's a win.

## 1. Dense FP8 GEMM — SM120 DeepGEMM tile port + `expected_m` decode hint — **LIVE**

File: `b12x/gemm/dense.py`. Verified present in the running image.

Ports DeepGEMM's SM120 FP8-GEMM tile strategy into b12x's CuTeDSL `dense_gemm`,
in two parts:

1. **M-independent default tile `(64,128)`**, replacing the prior static
   `(128,128)` pin. On wide-N shapes the `(128,128)` tile spans only
   `ceil(N/128)` column tiles (≤12 CTAs on 188 SMs → B-bandwidth-starved).
   `(64,128)` is the M-independent, prefill-safe choice and keeps b12x's
   one-kernel-per-`(N,K)` freeze/reuse contract.

2. **A DeepGEMM-style `expected_m` regime hint** in
   `_select_default_mma_tiler_mn(...)`. A caller (vLLM's decode forward sets
   `expected_m == live token count` under CUDA-graph capture) declares its batch
   regime and gets the regime-optimal MMA tile:
   - `expected_m ≤ 8  → (16,128)`   (decode)
   - `expected_m ≤ 128 → (32,128)`
   - `expected_m > 128 → (64,128)`   (prefill)

   It stays M-independent within a regime, so `(N,K,expected_m)` warms one kernel
   that serves every live M in that regime.

**Lossless** — tile selection only, byte-identical output (port validation
measured cos=1.0 vs the prior output and vs CUTLASS).

**Runs on:** DSV4-Flash's FP8 dense path, including the attention output (WO)
projection.

**Speed (port's own validation numbers, not a fresh A/B):** ~1.6× over FlashInfer
CUTLASS on the validation shape; the DSV4 decode WO projection measured ~2.5×
(71.7 → 29.1 µs) at the tile validated then. Caveat: the live decode tile has
since settled at `(16,128)` for `expected_m ≤ 8`, and we don't keep a baseline
image without the port, so a fresh before/after isn't available — treat these as
the port-validation figures, not a current measured delta.

Threaded through `dense_gemm`, `block_fp8_linear.py`, and `wo_projection.py`.

## 2. FC1 4-rows/warp retile — **ATTEMPTED, REVERTED (not on DSV4's path)**

File: `b12x/moe/fused/micro.py`.

Hey — I tried extending b12x's FC1 4-rows-per-warp retile to the NVFP4 decode
path. Honest status: **it's reverted.** The live gate is:

```python
# micro.py
if self.w4a16_mode and m == 1 and n <= 2048:
    # 4-rows/warp retile
```

The retile **gates on `w4a16_mode`**. DeepSeek-V4-Flash runs **NVFP4 experts**, so
`w4a16_mode` is `False` on its MoE path and **this retile never executes for
DSV4.** The NVFP4 extension I built was lossless (Estonia 30/30) but **lost ctx0
(~−2.5%)** at the perf gate, so it was reverted — there is no
`_MICRO_NVFP4_FC1_RETILE` in the live image. **Not re-applied.** Listed for
completeness; it is **not** a DSV4 speedup.
