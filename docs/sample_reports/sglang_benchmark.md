# sglang benchmark report

This is an example report shape for README/docs. Replace it with measured output from `uv run inference-lab bench-sglang`.

## Metadata

- `base_url`: `http://127.0.0.1:30000/v1`
- `model`: `Qwen/Qwen3-0.6B`
- `max_tokens`: `128`
- `stream`: `True`

## Summary

| scenario | concurrency | total | ok | errors | p50 latency (s) | p95 latency (s) | p50 TTFT (s) | p95 TTFT (s) | tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1 | 8 | 8 | 0 | 0.720000 | 0.910000 | 0.130000 | 0.180000 | 78.400000 |
| short | 4 | 8 | 8 | 0 | 1.120000 | 1.640000 | 0.190000 | 0.310000 | 156.200000 |
| long | 1 | 8 | 8 | 0 | 1.980000 | 2.410000 | 0.210000 | 0.290000 | 64.900000 |
| long | 4 | 8 | 8 | 0 | 3.360000 | 4.280000 | 0.330000 | 0.520000 | 121.700000 |
