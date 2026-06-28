# DeepSeek-V4-Flash + DSpark on the b12x vLLM fork (SM120 / RTX PRO 6000 Blackwell)
#
# Layers DeepSeek's DSpark speculative-decode vLLM integration on top of the
# public b12x runtime. Model weights are NOT baked in (hundreds of GB) — mount
# the base + DSpark-head checkpoints at runtime (see README.md).
#
# Build:
#   docker build -t deepspark:v4-flash .
# Run (single-user decode server on :9406):
#   see serve/serve_dsv4_flash_dspark.sh

ARG B12X_IMAGE=voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132
FROM ${B12X_IMAGE}

# The base image is the public b12x vLLM runtime; its tag (…-pr20-…) carries the
# DeepSeek DSpark speculative-decode integration. This layer adds only the
# reproducible serve config + Blackwell SM120 env defaults — it does NOT
# redistribute the fork source. (If you maintain local vLLM patches, mount them
# read-only at runtime over /opt/venv/lib/python3.12/site-packages/vllm.)

# Reproducible launch script.
COPY serve/serve_dsv4_flash_dspark.sh /usr/local/bin/serve_dsv4_flash_dspark.sh
RUN chmod +x /usr/local/bin/serve_dsv4_flash_dspark.sh

# Blackwell SM120 + b12x backend defaults (see LAUNCH_CONFIG.md for the full set).
ENV CUTE_DSL_ARCH=sm_120a \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_USE_B12X_MOE=1 \
    VLLM_USE_B12X_FP8_GEMM=1 \
    VLLM_USE_B12X_SPARSE_INDEXER=1 \
    VLLM_ENABLE_PCIE_ALLREDUCE=1 \
    VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
    B12X_MLA_SM120_UNIFIED=1 \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_LEVEL=SYS

# Models are mounted at runtime; this image carries only the serving stack.
# EXPOSE is informational — the serve script binds 0.0.0.0:9406.
EXPOSE 9406
