# Caveats — read before trusting any number

These results are honest but **distribution-, hardware-, and benchmark-specific.**
The most important caveat (ctx0 vs 128k) is first, because it is the most
counter-intuitive.

---

## Why ctx0 is slower than 128k

**The concern:** normally shorter context = faster decode (less KV to read), so
seeing **ctx0 (~235 t/s) slower than 128k (~258–283 t/s)** looks wrong — like a
measurement artifact.

**It is not a measurement artifact.** It is a real interaction of two things:
**sparse-MLA flattening the per-step cost** and **speculative-decoding acceptance
rising on the predictable long-context padding.** Here is the full decomposition.

### 1. What the benchmark actually times

Decode tok/s = `completion_tokens / decode_wall_time`, taken from the streamed
OpenAI-API usage. Prefill is timed **separately and excluded**. Both the ctx0 and
128k cells measure the same thing — pure post-prefill decode rate. The timer is
correct; prefill is **not** leaking into the decode number (measured prefill is
6.2–7.7k tok/s and lives in its own column).

### 2. Decode rate = acceptance ÷ per-step time

With speculative decoding:

```
decode tok/s  =  (mean accepted tokens per target step)  /  (per-step wall time)
```

Two levers: **acceptance** (a spec-decode property) and **per-step time** (the
target forward). The ctx0-vs-128k result is explained entirely by which lever
dominates.

### 3. Per-step time is ~flat across context — because of sparse MLA

The model runs **MLA + a sparse indexer** (`B12X_MLA_SPARSE` +
`VLLM_USE_B12X_SPARSE_INDEXER`). At decode, attention reads a **bounded top-k**
of the KV (a fixed budget selected by the indexer), **not** the full 128k. So the
classic "KV grows linearly → decode slows with context" effect is largely
**defeated**: the per-step forward at 128k is only marginally slower than at
ctx0, because attention is a small slice of the step and the **dense GEMM + MoE
dominate and are context-independent**. (NCU decode budget: dense ~33.7%, PCIe
all-reduce ~17.3%, MoE ~12.8%, MLA ~5.5%.)

### 4. Acceptance is *higher* at 128k — because the padding is predictable

This is the real driver. The benchmark builds long context from **calibrated
padding** (repetitive filler text). The `ignore_eos` continuation is then:

- **At ctx0:** conditioned on a tiny prompt → high-entropy, hard to predict → the
  DSpark draft is accepted less often (~2.84 of 5).
- **At 128k:** conditioned on 128k of **repetitive** padding → the continuation
  is strongly patterned and easy to predict → DSpark drafts it **more
  accurately** → higher acceptance → more accepted tokens per step.

The acceptance gain at 128k **more than offsets** the small per-step slowdown, so
128k posts a higher tok/s. That is the entire effect.

### 5. The caveats *on* this caveat — do not over-read "128k is faster"

- **It is distribution-flattered.** The synthetic padding is unusually
  predictable and **inflates long-context acceptance**. Real long-context text
  (e.g. dense legal documents) is far less repetitive, so a real workload would
  likely show 128k **at or below** ctx0, not above it. **The ctx0 number is the
  conservative, more transferable single-user figure.**
- **The 128k cell is noisy.** Look at the repeats:
  - ctx0 @ temp 0.1: `235.4, 233.5, 237.6` → spread ~4 (tight).
  - 128k @ temp 0.1: `297.5, 272.2, 280.7` → spread ~25.
  - 128k @ temp 0.0: `250.5, 231.0, 291.5` → spread ~60.
  Long-context acceptance varies run-to-run (which padding region the draft is
  sampling in, scheduler/cache timing), so the **128k mean has a wide error
  bar** while ctx0 is rock-solid.

**Bottom line:** ctx0 < 128k is genuine spec-decode behavior on this benchmark,
not a timing bug. **Trust ctx0 (~235, stable) as the headline single-user number;
read 128k as "at least as fast, often faster, but high-variance and flattered by
predictable padding."**

---

## Temperature

The often-cited **~225 t/s** baseline was at vLLM's **default `temperature=1.0`**
(the sustained harness previously sent no explicit temperature). Real production
sampling is **0.0–0.1**, where the stable ctx0 figure is **~235 t/s**. Lower temp
+ probabilistic draft raises acceptance legitimately. All headline numbers here
are at production temp; the temp is stated in every table.

---

## Losslessness

Every promoted change is **lossless**: greedy output is byte-identical to the
unmodified stack (verified with a 30/30 greedy parity suite, "Estonia"). Speed
changes that required any numeric reorder beyond the fp-association already in the
kernel were **rejected**, even when faster. Candidates that passed greedy parity
but regressed the sustained bench were also rejected. See [RESULTS.md](RESULTS.md)
§2.

---

## The honest kernel ceiling (300 t/s not reached)

A 300 t/s **ctx0** target was **not reached** by kernel work on this rig. The
cheap, lossless levers are consumed (split-K failed parity; swap-AB, committed-
prefix KV, max-active-clusters all regressed the canary; external FP4 does not
beat b12x; only a small MoE share-input win promoted). The honest ctx0 ceiling is
**~235–250 t/s**. Going higher needs either a **new kernel path**
(grouped/persistent dense decode GEMM, fused dequant/projection, head
replacement) **or higher draft acceptance** (a larger / tree draft head). Do not
read the 128k numbers as evidence the ctx0 ceiling is higher than it is.

---

## Single-rig, single-benchmark

- **One machine.** 4× RTX PRO 6000 Blackwell, PCIe Gen5, **no NVLink**, 300 W/GPU.
  TP all-reduce goes over PCIe (a one-shot b12x kernel); on an NVLink box the
  comm budget (~17% of decode) would differ and the balance could shift.
- **One benchmark distribution.** `llm_decode_bench` synthetic padding. Acceptance
  — and therefore tok/s — is distribution-dependent. Your workload will differ.
- **Single-user.** These are concurrency-1 decode rates. Throughput under load
  (multiple concurrent requests) is a different regime and not the subject here.
- **Third-party runtime.** The b12x fork is referenced, not re-licensed; the
  DSpark head and base model are DeepSeek releases. Reproduce against the linked
  public artifacts.
