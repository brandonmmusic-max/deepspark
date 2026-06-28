# Credits & acknowledgments

This artifact stands on open-source work by others.

## b12x vLLM — the runtime (open source)

The serving stack runs on the **b12x** vLLM fork (distributed as the
`voipmonitor/vllm` images on Docker Hub). It is **fully open source** and is
included here — both as the Docker runtime and as the vendored vLLM integration
source under [`overlay/vllm/`](overlay/vllm/) — full credit to voipmonitor and vllm team.

b12x provides the pieces that make this run on consumer Blackwell:

- the **SM120 CUTE kernels** (dense GEMM, MoE micro, fp8 GEMM),
- the **sparse-MLA decode** path + sparse indexer (`B12X_MLA_SPARSE`),
- the **PCIe one-shot all-reduce** (tensor-parallel without NVLink),
- the **DSpark speculative-decode** vLLM integration.

**Full credit for the b12x runtime and its kernels belongs to its author**, who
publishes it as the open-source `voipmonitor/vllm` images on Docker Hub.

## DeepSeek — the model & draft method

- **DeepSeek-V4-Flash** — the base model.
- **DSpark** — the speculative-decode draft head (DFlash backbone + Markov head +
  confidence head) and the V4-Flash-DSpark checkpoint.

## vLLM — upstream

The [vLLM](https://github.com/vllm-project/vllm) project (Apache-2.0), which b12x
forks.

## This repository

The benchmark results, launch configuration, caveats, and serve scripts here were
produced by Brandon Music on the hardware described in
[LAUNCH_CONFIG.md](LAUNCH_CONFIG.md).
