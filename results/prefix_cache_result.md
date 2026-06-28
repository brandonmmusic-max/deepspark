# vLLM Prefix Cache Result - dsv4-dspark-test :9406

Date: 2026-06-28

Scope honored: inspected and tested only the `dsv4-dspark-test` container on port `9406`. I did not touch `:9200`, `dsv4-9200-prod`, the official-base production launcher, power limits, or clocks.

## Bottom line

Prefix caching is enabled and working on this DSpark + chthonic b12x test stack.

It works alongside:

- DSpark speculative decode.
- `--attention-backend B12X_MLA_SPARSE`.
- `VLLM_USE_B12X_SPARSE_INDEXER=1`.
- fp8 KV cache / DeepSeek fp8 MLA KV format.
- The current launcher flag `--block-size 256`, with the runtime scheduler/cache metrics resolving their own prefix-cache granularity.

Recommendation: keep it enabled for the real long-document workload when the production serve is intentionally updated. It will not move decode tok/s, but it materially reduces repeated long-context prefill and TTFT. On this test stack, repeated shared-prefix requests cut prefill from about `1.52s` to `0.24s` in the 12.7k-token test, about `6.3x` faster, with a direct prefix-cache hit counter increment.

## Route note

The test container is `dsv4-dspark-test` and serves vLLM at the root of port `9406`, not under a URL subpath. Probes to the requested subpath returned 404:

```text
GET /dsv4-dspark-test/v1/models HTTP/1.1" 404 Not Found
GET /dsv4-dspark-test/metrics HTTP/1.1" 404 Not Found
```

The healthy test API was:

```text
http://127.0.0.1:9406/v1/models
model id: deepseek-v4-flash-dspark-test
```

## Step 1 - current status

`--enable-prefix-caching` is currently set in [zz_serve_dsv4_dspark_9406.sh](zz_serve_dsv4_dspark_9406.sh):

```text
--kv-cache-dtype fp8 --block-size 256 ...
--max-cudagraph-capture-size '$GRAPH_CAP' --attention-backend B12X_MLA_SPARSE \
--enable-chunked-prefill --enable-prefix-caching \
--speculative-config '{"method":"dspark", ...}'
```

Runtime confirms it is active. Startup log:

```text
non-default args: ... 'attention_backend': 'B12X_MLA_SPARSE', ... 'block_size': 256,
'kv_cache_dtype': 'fp8', 'enable_prefix_caching': True, ... 'speculative_config':
{'method': 'dspark', ...}
```

Engine config:

```text
Initializing a V1 LLM engine ... speculative_config=SpeculativeConfig(method='dspark', ...),
kv_cache_dtype=fp8, ... served_model_name=deepseek-v4-flash-dspark-test,
enable_prefix_caching=True, enable_chunked_prefill=True
```

Metrics expose and report prefix caching as active:

```text
vllm:cache_config_info{...,cache_dtype="fp8",enable_prefix_caching="True",
prefix_caching_hash_algo="sha256",user_specified_block_size="True"} 1.0
vllm:prefix_cache_queries_total ... 60531.0
vllm:prefix_cache_hits_total ... 36096.0
vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 36096.0
vllm:prompt_tokens_cached_total ... 36096.0
```

The metric label reports `block_size="4"` even though the CLI has `--block-size 256`; this comes from the runtime scheduler/hash block-size resolution for the actual KV cache groups. The B12X sparse MLA backend itself advertises `page_block_size == 64` for the kernel/page layout.

## Step 2 - measured behavior

### Shared-prefix cache-hit test

Prompt shape:

- Unique shared prefix: `12,690` tokens.
- Cold prompt A: `12,712` prompt tokens.
- Warm prompt B: `12,711` prompt tokens.
- A and B had the same long prefix and short differing suffixes.

Results:

| Request | TTFT | vLLM prefill time | local cache hit | local compute | prefix hit counter | spec decode |
|---|---:|---:|---:|---:|---:|---:|
| A cold | `1.540s` | `1.517s` | `0` | `12,712` | `0` | drafts `7`, draft tokens `35`, accepted `16` |
| B shared prefix | `0.268s` | `0.242s` | `12,544` | `167` | `12,544` | drafts `4`, draft tokens `20`, accepted `5` |

Speedup:

- Prefill: `1.516762 / 0.242400 = 6.26x`.
- TTFT: `1.539960 / 0.267724 = 5.75x`.
- Total request wall time: `1.626876 / 0.313782 = 5.18x`.

The cache evidence is not just timing: `vllm:prefix_cache_hits_total`, `vllm:prompt_tokens_cached_total`, and `prompt_tokens_by_source{source="local_cache_hit"}` all incremented by `12,544` on the second request, while computed prefill KV dropped to `167` tokens.

### Lossless same-prompt check

Prompt shape:

- Unique shared prefix: `11,172` tokens.
- Prompt: `11,196` tokens.
- Same prompt sent cold and then hot.
- Stop condition forced the answer to end at newline.

Results:

| Request | Output | TTFT | vLLM prefill time | local cache hit | local compute |
|---|---|---:|---:|---:|---:|
| cold | `4829` | `1.349s` | `1.328s` | `0` | `11,196` |
| hot same prompt | `4829` | `0.260s` | `0.239s` | `11,008` | `188` |

Lossless check: exact output match, `4829 == 4829`.

Speedup:

- Prefill: `1.327672 / 0.238682 = 5.56x`.
- TTFT: `1.348820 / 0.259755 = 5.19x`.

No runtime errors or tracebacks appeared during the tests. Recent logs did show expected first-shape JIT/cache messages, for example Triton JIT warnings and b12x `cute.compile` disk misses/hits, but not prefix-cache, sparse-indexer, fp8-KV, or spec-decode failures.

## Compatibility diagnosis

This stack does not reject prefix caching.

Relevant local code:

- `KVCacheManager.get_computed_blocks()` skips cache lookup only when prefix caching is disabled or the request is marked `skip_reading_prefix_cache`; otherwise it calls the coordinator and records stats. See `dspark_b12x_overlay/vllm/v1/core/kv_cache_manager.py:202-242`.
- The scheduler stores the cache-hit prefill stats as local cached tokens and external cached tokens. See `dspark_b12x_overlay/vllm/v1/core/sched/scheduler.py:633-731`.
- Metrics define and increment `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits`, `vllm:prompt_tokens_by_source`, and `vllm:prompt_tokens_cached`. See `dspark_b12x_overlay/vllm/v1/metrics/loggers.py:551-668` and `:1092-1168`.
- The B12X sparse MLA backend supports `fp8_ds_mla` and `fp8`, and its kernel block/page contract is `[64]`. See `dspark_b12x_overlay/vllm/v1/attention/backends/mla/b12x_mla_sparse.py:190-202`.
- The B12X sparse indexer path rejects FP4 indexer cache, not prefix caching; this run uses FP8 indexer cache. See `dspark_b12x_overlay/vllm/model_executor/layers/sparse_attn_indexer.py:785-790`.
- The generic MLA code disables prefix caching only for `TRITON_MLA` / `FLASHINFER` when `VLLM_BATCH_INVARIANT` is set; it does not include `B12X_MLA_SPARSE` in that disable list. See `dspark_b12x_overlay/vllm/model_executor/layers/attention/mla_attention.py:445-458`.
- DSpark spec decode has no prefix-cache rejection in its constructor path; the test metrics prove DSpark drafts/accepted tokens were produced during cache-hit requests. See `dspark_b12x_overlay/vllm/v1/spec_decode/dspark.py:21-50`.

The important incompatibility guard I found is DCP/hybrid-specific:

```text
disable_prefix_cache_for_dsv4_dcp =
  enable_caching and dcp_world_size > 1 and pcp_world_size == 1 and
  (is_deepseek_v4_hybrid_kv_cache_config(...) or _has_dcp_replicated)
```

When that flag is true, `cache_blocks()` returns immediately and `find_longest_cache_hit()` returns zero hits. See `dspark_b12x_overlay/vllm/v1/core/kv_cache_coordinator.py:507-529`, `:576-578`, and `:629-633`.

This run's engine config shows `decode_context_parallel_size=1`, so that DCP/hybrid kill switch is not active.

## How to enable if it is off

For this launcher, it is already enabled. If a future `:9406` test launcher is missing it, add this to the vLLM `serve` command:

```bash
--enable-prefix-caching
```

Keep the current verification surfaces:

```bash
curl -fsS http://127.0.0.1:9406/metrics | rg 'prefix_cache|prompt_tokens_cached|prompt_tokens_by_source|cache_config_info'
```

Expected active signals:

```text
vllm:cache_config_info{...,enable_prefix_caching="True",...} 1.0
vllm:prefix_cache_hits_total increments on repeated-prefix requests
vllm:prompt_tokens_by_source_total{source="local_cache_hit"} increments
vllm:request_prefill_kv_computed_tokens_sum is much smaller on the hot request
```

Do not expect decode tok/s to improve. Prefix caching skips repeated prefill KV computation; it does not make the decode kernel faster.

## Recommendation for Brandon workload

Turn it on for repeated long legal-doc / KG-context workloads, subject to the same test-stack constraints above. The measured effect is exactly the desired one: large shared-prefix prompts hit cached KV, compute only the short differing suffix, and cut TTFT/E2E latency by roughly `5x-6x` in these tests.

Do not enable blindly if changing context parallelism or KV group topology. In particular, DCP > 1 with DeepSeek V4 hybrid / DFlash-replicated draft groups takes the code path that disables prefix cache hits.
