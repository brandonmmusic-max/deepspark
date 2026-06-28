# Launch configuration — full stack

The exact configuration behind every number in this repo. The authoritative
script is [serve/serve_dsv4_flash_dspark.sh](serve/serve_dsv4_flash_dspark.sh);
this document explains it.

---

## Hardware

| Component | Detail |
|---|---|
| GPU | 4× NVIDIA RTX PRO 6000 Blackwell, 96 GB GDDR7 each (GPU0/2 Max-Q, GPU1/3 Workstation) |
| Compute | SM120 (Blackwell), compiled `CUTE_DSL_ARCH=sm_120a` |
| Interconnect | PCIe Gen5 ×16 per GPU, **no NVLink** |
| CPU | 48 cores |
| RAM | 512 GB DDR5, 8-channel (~307 GB/s) |
| Driver | 595.58.03 |
| CUDA | 13.2 (image tag `…-cu132`) |

### Power constraints (held constant — not tuned)

All four GPUs run at a **300 W power cap**:

```
GPU0  RTX PRO 6000 Blackwell Max-Q       limit 300W  (default 300, max 300)
GPU1  RTX PRO 6000 Blackwell Workstation  limit 300W  (default 600, max 600)
GPU2  RTX PRO 6000 Blackwell Max-Q       limit 300W  (default 300, max 300)
GPU3  RTX PRO 6000 Blackwell Workstation  limit 300W  (default 600, max 600)
```

The two Workstation cards are **power-limited down to 300 W** to match the two
Max-Q cards, so the box runs a uniform **300 W/GPU (~1200 W aggregate for the 4
GPUs)**. **No `nvidia-smi -pl` and no clock changes were made for any run** — a
locked-clock control run reproduced the same ctx0/128k ordering, ruling out a
clock/thermal artifact. All numbers are at stock 300 W.

---

## Software stack

| Layer | Value |
|---|---|
| Runtime image | `voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132` (public b12x vLLM fork) |
| vLLM | b12x fork + **DSpark speculative-decode overlay** (`./overlay/vllm`, mounted RO) |
| Attention backend | `B12X_MLA_SPARSE` (sparse-MLA decode + sparse indexer) |
| MoE / linear backend | `b12x` |
| Model runner | V2 (`VLLM_USE_V2_MODEL_RUNNER=1`) |
| KV cache | fp8 |

---

## Model

| Role | Path / repo | Key config |
|---|---|---|
| Base | `deepseek-v4-flash-official` (HF `deepseek-ai/DeepSeek-V4-Flash`) | `DeepseekV4ForCausalLM`, hidden 4096, vocab 129280, NVFP4 weights |
| Draft head | `DeepSeek-V4-Flash-DSpark` (HF) | DSpark block5 (γ=5): DFlash backbone + Markov head (rank 256) + confidence head; KV-inject target layers 40/41/42 |

---

## The serve command

```bash
python -m vllm.entrypoints.cli.main serve <BASE_MODEL> \
  --served-model-name deepseek-v4-flash-dspark-test --host 0.0.0.0 --port 9406 \
  --kv-cache-dtype fp8 --block-size 256 --load-format safetensors \
  --tensor-parallel-size 4 --moe-backend b12x --linear-backend b12x \
  --gpu-memory-utilization 0.89 --max-model-len 393216 --max-num-seqs 64 \
  --async-scheduling --no-scheduler-reserve-full-isl --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 64 --attention-backend B12X_MLA_SPARSE \
  --enable-chunked-prefill --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=high \
  --enable-flashinfer-autotune \
  --speculative-config '{
      "method":"dspark",
      "model":"<DRAFT_HEAD>",
      "num_speculative_tokens":5,
      "draft_sample_method":"probabilistic",
      "moe_backend":"b12x",
      "use_local_argmax_reduction":false,
      "dspark_confidence_schedule":false,
      "dspark_sts_temperatures":[1.15,1.3,1.45,1.6,1.75],
      "dspark_scheduler_knee_tokens":64.0,
      "dspark_scheduler_cost_exponent":2.0
  }'
```

### Flags that matter most

| Flag | Why |
|---|---|
| `--attention-backend B12X_MLA_SPARSE` | Sparse-MLA decode: attends a bounded top-k of KV, not the full context — this is why per-step decode cost is ~flat across context length |
| `--speculative-config method=dspark, num_speculative_tokens=5` | DSpark draft, γ=5 |
| `draft_sample_method=probabilistic` | Probabilistic (not argmax) draft sampling — raises acceptance legitimately at low temp |
| `--kv-cache-dtype fp8` | fp8 MLA KV (≈1.47× cache vs bf16) |
| `--tensor-parallel-size 4` | TP across 4 GPUs; all-reduce over **PCIe** (no NVLink) |
| `--gpu-memory-utilization 0.89` / `--max-model-len 393216` | 384k context budget |
| `--enable-prefix-caching` | Shared-prefix KV reuse (see RESULTS §3); keep DCP=1 to retain hits |
| `--max-cudagraph-capture-size 64` + `cudagraph_mode=FULL_AND_PIECEWISE` | Full CUDA-graph decode |

### Key environment (b12x backends)

```
CUTE_DSL_ARCH=sm_120a
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_FP8_GEMM=1
VLLM_USE_B12X_MHC=1
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_ENABLE_PCIE_ALLREDUCE=1
VLLM_PCIE_ALLREDUCE_BACKEND=b12x      # one-shot PCIe all-reduce (no NVLink)
B12X_MLA_SM120_UNIFIED=1
VLLM_USE_FLASHINFER_SAMPLER=1
NCCL_P2P_LEVEL=SYS  NCCL_IB_DISABLE=1  # PCIe-only topology
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Optional / default-off tuning knobs present in the script (kept off for the
headline numbers): `B12X_DENSE_SPLITK*`, `B12X_DENSE_MXFP8_SWAP_AB`,
`B12X_MICRO_MAX_ACTIVE_CLUSTERS`, `DSPARK_CONFIDENCE_SCHEDULE`,
`DSPARK_COMMITTED_PREFIX_KV`. See [RESULTS.md](RESULTS.md) §2 for which of these
walled vs promoted.

---

## Isolation note

This stack runs on port **9406** as `dsv4-dspark-test`, fully isolated from any
production server. The launch script hard-refuses to bind `:9200` or the prod
container name, and mounts the base model **read-only**.
