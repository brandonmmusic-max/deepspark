#!/usr/bin/env bash
# Isolated DeepSeek-V4-Flash DSpark test server.
# Does not touch :9200 or dsv4-9200-prod; mounts the official base read-only.
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132}"
NAME="${NAME:-dsv4-dspark-test}"
BASE_MODEL_HOST_DIR="${BASE_MODEL_HOST_DIR:-/home/brandonmusic/models}"
BASE_MODEL_DIR_NAME="${BASE_MODEL_DIR_NAME:-deepseek-v4-flash-official}"
HEAD_MODEL_HOST_DIR="${HEAD_MODEL_HOST_DIR:-/media/brandonmusic/nvme0n1p3/models}"
HEAD_MODEL_DIR_NAME="${HEAD_MODEL_DIR_NAME:-DeepSeek-V4-Flash-DSpark-head}"
MODEL_PATH="${MODEL_PATH:-/models-base/$BASE_MODEL_DIR_NAME}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models-head/$HEAD_MODEL_DIR_NAME}"
SERVED_NAME="${SERVED_NAME:-deepseek-v4-flash-dspark-test}"
PORT="${PORT:-9406}"
TP="${TP:-4}"
GPU_UTIL="${GPU_UTIL:-0.89}"
MAXLEN="${MAXLEN:-393216}"
MAX_SEQS="${MAX_SEQS:-64}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
GRAPH_CAP="${GRAPH_CAP:-64}"
CACHE_DIR="${CACHE_DIR:-/home/brandonmusic/.cache/dsv4-dspark-test}"
VLLM_OVERLAY="${VLLM_OVERLAY:-/home/brandonmusic/klc-linux/dspark_b12x_overlay/vllm}"
B12X_OVERLAY="${B12X_OVERLAY:-}"
DSPARK_NUM_SPECULATIVE_TOKENS="${DSPARK_NUM_SPECULATIVE_TOKENS:-5}"
NSYS_PROFILE="${NSYS_PROFILE:-0}"
NSYS_LAUNCH_SESSION="${NSYS_LAUNCH_SESSION:-0}"
NSYS_SESSION_NAME="${NSYS_SESSION_NAME:-dspark_9406_decode}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-node}"
NSYS_OUTPUT="${NSYS_OUTPUT:-/cache/nsys/dspark_profile}"
NSYS_DURATION="${NSYS_DURATION:-240}"
DSPARK_CONFIDENCE_SCHEDULE="${DSPARK_CONFIDENCE_SCHEDULE:-false}"
if [ -z "${DSPARK_STS_TEMPS:-}" ]; then
  case "$DSPARK_NUM_SPECULATIVE_TOKENS" in
    3) DSPARK_STS_TEMPS="[1.15,1.3,1.45]" ;;
    4) DSPARK_STS_TEMPS="[1.15,1.3,1.45,1.6]" ;;
    5) DSPARK_STS_TEMPS="[1.15,1.3,1.45,1.6,1.75]" ;;
    6) DSPARK_STS_TEMPS="[1.15,1.3,1.45,1.6,1.75,1.9]" ;;
    7) DSPARK_STS_TEMPS="[1.15,1.3,1.45,1.6,1.75,1.9,2.05]" ;;
    *) echo "FATAL: set DSPARK_STS_TEMPS for DSPARK_NUM_SPECULATIVE_TOKENS=$DSPARK_NUM_SPECULATIVE_TOKENS"; exit 1 ;;
  esac
fi
DSPARK_SCHEDULER_KNEE_TOKENS="${DSPARK_SCHEDULER_KNEE_TOKENS:-64.0}"
DSPARK_SCHEDULER_COST_EXPONENT="${DSPARK_SCHEDULER_COST_EXPONENT:-2.0}"
DSPARK_COMMITTED_PREFIX_KV="${DSPARK_COMMITTED_PREFIX_KV:-0}"
VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE="${VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE:-0}"
VLLM_B12X_MLA_SPEC_DECODE_MAX_Q="${VLLM_B12X_MLA_SPEC_DECODE_MAX_Q:-8}"
B12X_DENSE_SPLITK_TURBO="${B12X_DENSE_SPLITK_TURBO:-1}"
B12X_DENSE_SPLITK="${B12X_DENSE_SPLITK:-0}"
B12X_DENSE_SPLITK_LOSSLESS="${B12X_DENSE_SPLITK_LOSSLESS:-1}"
B12X_DENSE_SPLITK_MAX="${B12X_DENSE_SPLITK_MAX:-2}"
B12X_DENSE_SPLITK_WAVE="${B12X_DENSE_SPLITK_WAVE:-0}"
B12X_DENSE_SPLITK_WAVE_MAX="${B12X_DENSE_SPLITK_WAVE_MAX:-8}"
B12X_DENSE_MXFP8_SWAP_AB="${B12X_DENSE_MXFP8_SWAP_AB:-1}"
B12X_DENSE_MXFP8_SWAP_TILE_N="${B12X_DENSE_MXFP8_SWAP_TILE_N:-32}"
B12X_MICRO_MAX_ACTIVE_CLUSTERS="${B12X_MICRO_MAX_ACTIVE_CLUSTERS:-}"
if [ "$DSPARK_CONFIDENCE_SCHEDULE" = "true" ]; then
  DSPARK_VERIFY_LEN_BUCKETS="${DSPARK_VERIFY_LEN_BUCKETS:-6}"
  DSPARK_VERIFY_BUCKET_REQ_COUNTS="${DSPARK_VERIFY_BUCKET_REQ_COUNTS:-1,2,4,8}"
else
  DSPARK_VERIFY_LEN_BUCKETS="${DSPARK_VERIFY_LEN_BUCKETS:-6}"
  DSPARK_VERIFY_BUCKET_REQ_COUNTS="${DSPARK_VERIFY_BUCKET_REQ_COUNTS:-1,2,4,8}"
fi
mkdir -p "$CACHE_DIR"

echo "== dspark preflight =="
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "FATAL: image not present: $IMAGE"
  exit 1
}
[ "$PORT" != "9200" ] || {
  echo "FATAL: refusing to use production port 9200"
  exit 1
}
[ "$NAME" != "dsv4-9200-prod" ] || {
  echo "FATAL: refusing to use production container name dsv4-9200-prod"
  exit 1
}
[ -d "$BASE_MODEL_HOST_DIR/$BASE_MODEL_DIR_NAME" ] || {
  echo "FATAL: base model missing: $BASE_MODEL_HOST_DIR/$BASE_MODEL_DIR_NAME"
  exit 1
}
[ -d "$HEAD_MODEL_HOST_DIR/$HEAD_MODEL_DIR_NAME" ] || {
  echo "FATAL: DSpark head missing: $HEAD_MODEL_HOST_DIR/$HEAD_MODEL_DIR_NAME"
  exit 1
}
[ -d "$VLLM_OVERLAY" ] || {
  echo "FATAL: patched vLLM overlay missing: $VLLM_OVERLAY"
  exit 1
}
EXTRA_MOUNTS=()
EXTRA_ENVS=()
if [ -n "$B12X_OVERLAY" ]; then
  [ -d "$B12X_OVERLAY" ] || {
    echo "FATAL: patched b12x overlay missing: $B12X_OVERLAY"
    exit 1
  }
  EXTRA_MOUNTS+=(
    -v "$B12X_OVERLAY":/opt/venv/lib/python3.12/site-packages/b12x:ro
  )
fi
if [ -n "$B12X_MICRO_MAX_ACTIVE_CLUSTERS" ]; then
  EXTRA_ENVS+=(-e B12X_MICRO_MAX_ACTIVE_CLUSTERS="$B12X_MICRO_MAX_ACTIVE_CLUSTERS")
fi
for shard in 46 47 48; do
  shard_file="$HEAD_MODEL_HOST_DIR/$HEAD_MODEL_DIR_NAME/model-000${shard}-of-00048.safetensors"
  [ -s "$shard_file" ] || {
    echo "FATAL: DSpark head shard missing or empty: $shard_file"
    exit 1
  }
done
for required in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json; do
  [ -s "$HEAD_MODEL_HOST_DIR/$HEAD_MODEL_DIR_NAME/$required" ] || {
    echo "FATAL: DSpark head metadata missing: $required"
    exit 1
  }
done
echo "image=$IMAGE base=$BASE_MODEL_HOST_DIR/$BASE_MODEL_DIR_NAME draft=$HEAD_MODEL_HOST_DIR/$HEAD_MODEL_DIR_NAME port=$PORT graph_cap=$GRAPH_CAP cache=$CACHE_DIR vllm_overlay=$VLLM_OVERLAY b12x_overlay=${B12X_OVERLAY:-<image>} dspark_num_speculative_tokens=$DSPARK_NUM_SPECULATIVE_TOKENS dspark_confidence_schedule=$DSPARK_CONFIDENCE_SCHEDULE dspark_committed_prefix_kv=$DSPARK_COMMITTED_PREFIX_KV verify_len_buckets=$DSPARK_VERIFY_LEN_BUCKETS verify_bucket_req_counts=$DSPARK_VERIFY_BUCKET_REQ_COUNTS spec_extend_as_decode=$VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE dense_splitk=$B12X_DENSE_SPLITK dense_splitk_lossless=$B12X_DENSE_SPLITK_LOSSLESS dense_splitk_max=$B12X_DENSE_SPLITK_MAX dense_splitk_turbo=$B12X_DENSE_SPLITK_TURBO dense_splitk_wave=$B12X_DENSE_SPLITK_WAVE dense_splitk_wave_max=$B12X_DENSE_SPLITK_WAVE_MAX dense_mxfp8_swap_ab=$B12X_DENSE_MXFP8_SWAP_AB dense_mxfp8_swap_tile_n=$B12X_DENSE_MXFP8_SWAP_TILE_N micro_max_active_clusters=${B12X_MICRO_MAX_ACTIVE_CLUSTERS:-<unset>} nsys_profile=$NSYS_PROFILE nsys_launch_session=$NSYS_LAUNCH_SESSION nsys_session_name=$NSYS_SESSION_NAME nsys_cuda_graph_trace=$NSYS_CUDA_GRAPH_TRACE nsys_output=$NSYS_OUTPUT nsys_duration=$NSYS_DURATION"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --gpus all --runtime nvidia --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --restart unless-stopped \
  -v "$BASE_MODEL_HOST_DIR":/models-base:ro \
  -v "$HEAD_MODEL_HOST_DIR":/models-head:ro \
  -v "$CACHE_DIR":/cache \
  -v "$VLLM_OVERLAY":/opt/venv/lib/python3.12/site-packages/vllm:ro \
  "${EXTRA_MOUNTS[@]}" \
  "${EXTRA_ENVS[@]}" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUTE_DSL_ARCH=sm_120a \
  -e XDG_CACHE_HOME=/cache/jit -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions -e TRITON_CACHE_DIR=/cache/jit/triton -e VLLM_CACHE_DIR=/cache/jit/vllm -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/b12x/cute_compile \
  -e HF_HUB_OFFLINE=1 -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 -e B12X_MHC_MAX_TOKENS=16384 -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_USE_B12X_MHC=1 -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=$([ "$MOE_BACKEND" = "b12x" ] && echo 1 || echo 0) \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE="$VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE" \
  -e VLLM_B12X_MLA_SPEC_DECODE_MAX_Q="$VLLM_B12X_MLA_SPEC_DECODE_MAX_Q" \
  -e VLLM_SPEC_DECODE_QUERY_LEN_BUCKETS="$DSPARK_VERIFY_LEN_BUCKETS" -e VLLM_SPEC_DECODE_BUCKET_REQ_COUNTS="$DSPARK_VERIFY_BUCKET_REQ_COUNTS" -e DSPARK_VERIFY_LEN_BUCKETS="$DSPARK_VERIFY_LEN_BUCKETS" \
  -e DSPARK_COMMITTED_PREFIX_KV="$DSPARK_COMMITTED_PREFIX_KV" \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e B12X_MLA_SM120_UNIFIED=1 -e USES_B12X=True -e B12X_DENSE_SPLITK_TURBO="$B12X_DENSE_SPLITK_TURBO" -e B12X_DENSE_SPLITK="$B12X_DENSE_SPLITK" -e B12X_DENSE_SPLITK_LOSSLESS="$B12X_DENSE_SPLITK_LOSSLESS" -e B12X_DENSE_SPLITK_MAX="$B12X_DENSE_SPLITK_MAX" -e B12X_DENSE_SPLITK_WAVE="$B12X_DENSE_SPLITK_WAVE" -e B12X_DENSE_SPLITK_WAVE_MAX="$B12X_DENSE_SPLITK_WAVE_MAX" -e B12X_DENSE_MXFP8_SWAP_AB="$B12X_DENSE_MXFP8_SWAP_AB" -e B12X_DENSE_MXFP8_SWAP_TILE_N="$B12X_DENSE_MXFP8_SWAP_TILE_N" -e B12X_W4A16_TC_DECODE=1 \
  "$IMAGE" \
  /bin/bash -lc "
    set -euo pipefail
    unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
    cd /
    mkdir -p /cache/nsys
    SERVE_CMD=(/opt/venv/bin/python -m vllm.entrypoints.cli.main serve '$MODEL_PATH' \
      --served-model-name '$SERVED_NAME' --host 0.0.0.0 --port '$PORT' \
      --kv-cache-dtype fp8 --block-size 256 --load-format safetensors \
      --tensor-parallel-size '$TP' --moe-backend '$MOE_BACKEND' --linear-backend b12x \
      --gpu-memory-utilization '$GPU_UTIL' --max-model-len '$MAXLEN' --max-num-seqs '$MAX_SEQS' \
      --async-scheduling --no-scheduler-reserve-full-isl --max-num-batched-tokens '$MAX_BATCHED' \
      --max-cudagraph-capture-size '$GRAPH_CAP' --attention-backend B12X_MLA_SPARSE \
      --enable-chunked-prefill --enable-prefix-caching \
      --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}' \
      --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
      --reasoning-parser deepseek_v4 \
      --default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=high \
      --enable-flashinfer-autotune \
      --speculative-config '{\"method\":\"dspark\",\"model\":\"$DRAFT_MODEL_PATH\",\"num_speculative_tokens\":$DSPARK_NUM_SPECULATIVE_TOKENS,\"draft_sample_method\":\"probabilistic\",\"moe_backend\":\"$MOE_BACKEND\",\"use_local_argmax_reduction\":false,\"dspark_confidence_schedule\":$DSPARK_CONFIDENCE_SCHEDULE,\"dspark_sts_temperatures\":$DSPARK_STS_TEMPS,\"dspark_scheduler_knee_tokens\":$DSPARK_SCHEDULER_KNEE_TOKENS,\"dspark_scheduler_cost_exponent\":$DSPARK_SCHEDULER_COST_EXPONENT}')
	    if [ '$NSYS_PROFILE' = '1' ]; then
	      exec /usr/local/bin/nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace='$NSYS_CUDA_GRAPH_TRACE' --sample=none --cpuctxsw=none --duration='$NSYS_DURATION' --wait=primary --force-overwrite=true --output='$NSYS_OUTPUT' \"\${SERVE_CMD[@]}\"
	    fi
	    if [ '$NSYS_LAUNCH_SESSION' = '1' ]; then
	      exec /usr/local/bin/nsys launch --session-new='$NSYS_SESSION_NAME' --trace=cuda,nvtx,osrt --cuda-graph-trace='$NSYS_CUDA_GRAPH_TRACE' --show-output=true --wait=all \"\${SERVE_CMD[@]}\"
	    fi
	    exec \"\${SERVE_CMD[@]}\"
	  "
echo "Launched $NAME (DSpark isolated test: port=$PORT model=$SERVED_NAME)"
