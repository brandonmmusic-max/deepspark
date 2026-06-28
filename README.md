# DeepSeek-V4-Flash + DSpark on b12x vLLM (SM120 / RTX PRO 6000 Blackwell)

Single-user speculative decoding for **DeepSeek-V4-Flash** using DeepSeek's
**DSpark** draft head on the **b12x** vLLM fork, served on a 4× **RTX PRO 6000
Blackwell** workstation (SM120, PCIe Gen5, **no NVLink**).

This repo documents — reproducibly — the serving stack, the **full `llm_decode_bench`
results**, the launch configuration, the hardware/power constraints, and a
careful set of **caveats** (including why 0-context decode measures *slower* than
128k-context decode on this benchmark).

> **Honesty note.** Numbers here are the standard `llm_decode_bench` sustained
> single-user decode rate, measured byte-identical, lossless. No reframed or
> "thinking-token" harness. Where the target was not met, this README says so.

---

## Headline results (single-user decode, lossless)

| Context | Decode tok/s (temp 0.1) | Decode tok/s (temp 0.0) |
|--------:|------------------------:|------------------------:|
| ctx 0   | **235.5**               | 234.0                   |
| 128k    | **283.5**               | 257.7                   |

- Draft: DSpark head, `num_speculative_tokens=5`, `draft_sample_method=probabilistic`.
- Mean accepted length at ctx0 ≈ **2.84 / 5**.
- Original baseline often cited as ~225 t/s was measured at vLLM's **default
  `temperature=1.0`**; at the real production temperature (0.0–0.1) the stable
  ctx0 figure is **~235 t/s**.

**Verdict (honest):** a 300 t/s single-user **ctx0** target was **not reached**
by kernel-level work on this rig. The decode kernels (dense GEMM, MoE micro,
sparse-MLA, PCIe all-reduce) are essentially tapped at **~235–250 t/s ctx0**; the
only remaining *multiplicative* lever is **draft acceptance** (a larger / tree
draft head). See [RESULTS.md](RESULTS.md) and [CAVEATS.md](CAVEATS.md).

---

## Why is ctx0 *slower* than 128k? (short version)

It is **not a measurement artifact.** The 128k per-step forward is genuinely
~19% slower than ctx0 (longer context costs more, as expected) — but
speculative-decoding **acceptance** is far higher on the predictable 128k padding
(mean accepted **3.22 vs 1.87** of 5, **+47% tokens/step**), and that overwhelms
the slower forward to net **+24% tok/s**. It is distribution-flattered and the
128k cell is noisy, so trust ctx0 (~235, rock-stable) as the conservative
single-user number. Measured breakdown:
[Measured acceptance](#measured-acceptance--the-evidence) · full analysis:
[CAVEATS.md](CAVEATS.md#why-ctx0-is-slower-than-128k).

---

## Measured acceptance — the evidence

The explanation above is backed by the live vLLM spec-decode counters
(`vllm:spec_decode_num_accepted_tokens_per_pos_total`), captured as the delta
across sustained `llm_decode_bench` runs at production config (γ=5, temp 0.1) —
measured, not asserted.

| Draft token position | Acceptance @ ctx0 | Acceptance @ 128k |
|:---|---:|---:|
| 1 | 74.0% | 86.3% |
| 2 | 50.4% | 74.8% |
| 3 | 32.2% | 65.4% |
| 4 | 19.4% | 52.1% |
| 5 | 10.9% | 43.5% |
| **Mean accepted / 5** | **1.87** | **3.22** |
| **Emitted per step** (incl. bonus) | **2.87** | **4.22** |
| **Decode tok/s** | **237.7** | **293.9** |

**This is the whole story of why ctx0 < 128k — and it confirms that a longer
context really is slower per step:**

- Implied per-step forward time is **12.1 ms @ ctx0 vs 14.4 ms @ 128k** — so the
  128k forward *is* genuinely **~19% slower** (longer context costs more per step,
  exactly as expected; sparse-MLA softens that cost but does not erase it).
- But the draft lands **far more often** on the predictable long-context padding —
  mean accepted **3.22 vs 1.87**, i.e. **+47% tokens emitted per step**.
- Acceptance (+47%) overwhelms the slower forward (+19%), netting **+24% decode
  tok/s** (293.9 vs 237.7).

So `ctx0 < 128k` is a **speculative-acceptance effect riding on top of a
(correctly) slower per-step forward** — not a measurement artifact. Caveat: the
128k acceptance is **inflated by the benchmark's repetitive synthetic padding**;
real long-context text is less predictable, so treat ctx0 (~235) as the
conservative single-user number. The steep ctx0 decay (token 5 lands only ~11%)
is also why lengthening the draft chain has little headroom on this head.

---

## Hardware & power

| Component | Spec |
|---|---|
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell, 96 GB each (2× Workstation, 2× Max-Q) |
| Arch | SM120 (Blackwell), `CUTE_DSL_ARCH=sm_120a` |
| Interconnect | PCIe Gen5 ×16, **no NVLink** (TP all-reduce over PCIe via b12x one-shot) |
| **Power cap** | **300 W per GPU** (all 4), ~1200 W aggregate. Not changed during any run. |
| CPU / RAM | 48 cores / 512 GB DDR5 (8-channel, ~307 GB/s) |
| Driver / CUDA | 595.58.03 / CUDA 13.2 |

Power limits were **held at 300 W/GPU throughout** — no `nvidia-smi -pl`, no clock
changes. Results are what this stack delivers at stock power.

---

## Reproduce it

You need three things: the **runtime image**, the **two model checkpoints**, and
the **serve script** in this repo.

### 1. Runtime image

Either pull the prebuilt image:

```bash
docker pull verdictai/deepspark:v4-flash          # hub.docker.com/r/verdictai/deepspark
```

…or build it yourself on top of the public b12x base:

```bash
docker build -t deepspark:v4-flash .
```

The base is the public `voipmonitor/vllm:chthonic-…-pr20-cu132` **b12x** runtime —
an open-source vLLM fork (see [CREDITS.md](CREDITS.md)). This repo vendors the
matching b12x vLLM + DSpark integration source under [`overlay/vllm/`](overlay/vllm/)
and the [Dockerfile](Dockerfile) layers it in, so the build reproduces the exact
serving stack used for every number here.

### 2. Model checkpoints (Hugging Face)

| Role | Repo |
|---|---|
| Base | `deepseek-ai/DeepSeek-V4-Flash` (served as `deepseek-v4-flash-official`) |
| Draft head | `deepseek-ai/DeepSeek-V4-Flash-DSpark` (block5, γ=5) |

Download both, then point the serve script at them (mounted read-only).

### 3. Serve + benchmark

```bash
# Launch (edit the model paths at the top for your machine):
bash serve/serve_dsv4_flash_dspark.sh

# Benchmark single-user decode (standard llm_decode_bench):
#   https://github.com/<llm-inference-bench>  (see bench/README.md)
python llm_decode_bench.py \
  --base-url http://127.0.0.1:9406 \
  --model deepseek-v4-flash-dspark-test \
  --context 0 --temperature 0.1 --ignore-eos --sustained
```

Full flag-by-flag launch config: [LAUNCH_CONFIG.md](LAUNCH_CONFIG.md).
Exact bench invocation + context sweep: [bench/README.md](bench/README.md).

---

## What's in here

| File | Contents |
|---|---|
| [RESULTS.md](RESULTS.md) | Full `llm_decode_bench` grid: production-temp baseline, the optimization-lever campaign (what walled, what promoted), prefix-cache results, raw repeats |
| [LAUNCH_CONFIG.md](LAUNCH_CONFIG.md) | The complete serve command, every flag explained, hardware, power, software versions |
| [CAVEATS.md](CAVEATS.md) | ctx0-vs-128k analysis, synthetic-bench limits, run-to-run noise, temperature, losslessness, the honest kernel ceiling |
| [Dockerfile](Dockerfile) | Reproducible image: public b12x base + DSpark vLLM overlay |
| [serve/](serve/) | The actual launch script used for every number in this repo |
| [overlay/vllm/](overlay/vllm/) | Vendored b12x vLLM + DSpark integration source (open source, credited) — what the image layers in |
| [CREDITS.md](CREDITS.md) | Attribution: b12x author, DeepSeek, vLLM |
| [results/](results/) | Raw campaign result docs (unedited) |

---

## Acknowledgments

The serving stack runs on the open-source **b12x vLLM fork** (the
`voipmonitor/vllm` runtime) — its SM120 CUTE kernels, sparse-MLA decode, PCIe
all-reduce, and DSpark integration are what make this work on consumer Blackwell.
It is included and documented here **with the author's permission and at their
request**; full credit for the runtime belongs to its author. **DeepSeek**
released DeepSeek-V4-Flash and the DSpark draft method. Full attribution:
[CREDITS.md](CREDITS.md).

## Scope / disclaimer

This is a **single-rig research artifact**, not a product. The b12x runtime is an
open-source vLLM fork, shared with permission and credited above. DSpark and
DeepSeek-V4-Flash are DeepSeek releases. Numbers are specific to this hardware,
this driver/CUDA, and this benchmark distribution; treat them as a reproducible
data point, not a universal claim.
