# Benchmark — how the numbers were measured

All decode numbers in [../RESULTS.md](../RESULTS.md) are from the standard
**`llm_decode_bench`** harness: sustained single-user decode throughput,
`ignore_eos`, timed from the streamed OpenAI-API usage, lossless.

## Invocation

With the server up on `:9406` (see [../serve/serve_dsv4_flash_dspark.sh](../serve/serve_dsv4_flash_dspark.sh)):

```bash
python llm_decode_bench.py \
  --base-url http://127.0.0.1:9406 \
  --model deepseek-v4-flash-dspark-test \
  --context 0 \
  --temperature 0.1 \
  --ignore-eos \
  --sustained
```

Sweep context to reproduce the grid:

```bash
for CTX in 0 16384 32768 65536 131072; do
  python llm_decode_bench.py --base-url http://127.0.0.1:9406 \
    --model deepseek-v4-flash-dspark-test \
    --context $CTX --temperature 0.1 --ignore-eos --sustained
done
```

## Important measurement rules

1. **Set `--temperature` explicitly.** Omitting it falls back to vLLM's default
   `temperature=1.0`, which is *not* the production sampling temp and gives a
   different (lower, ~225) number. Production temp is 0.0–0.1.
2. **`--ignore-eos`** so decode runs for a fixed length and you measure sustained
   rate, not until-EOS.
3. **Repeat ≥3×**, especially at long context — the 128k cell is high-variance
   (see [../CAVEATS.md](../CAVEATS.md)). Report the mean and the individual
   repeats.
4. **Decode only.** Prefill is a separate measurement; do not fold it into the
   decode tok/s.
5. **Lossless check separately.** Greedy (`temperature=0`) output must be
   byte-identical to the unmodified stack before trusting any speed delta.

## The harness

`llm_decode_bench.py` is part of a standard LLM-inference benchmark suite (not
vendored here to avoid re-licensing a third-party tool). Any OpenAI-compatible
sustained-decode benchmark that (a) sends an explicit temperature, (b) uses
`ignore_eos`, and (c) times decode from streamed usage will reproduce these
numbers against the same server + models.
