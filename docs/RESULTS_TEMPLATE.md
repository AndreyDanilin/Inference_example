# Inference Lab Results

Date:

## Hardware

- GPU:
- VRAM:
- Driver:
- CUDA:
- OS:
- Python:

## SGLang Serving

- Model:
- Config:
- Max tokens:
- Concurrency:

| scenario | concurrency | total | ok | errors | p50 latency (s) | p95 latency (s) | p50 TTFT (s) | p95 TTFT (s) | tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1 |  |  |  |  |  |  |  |  |
| short | 4 |  |  |  |  |  |  |  |  |
| long | 1 |  |  |  |  |  |  |  |  |
| long | 4 |  |  |  |  |  |  |  |  |

## Optimization Progression

Include generated SVG charts from:

```bash
uv run inference-lab plot-optimization --input reports/optimization_steps.yaml --output-dir reports/optimization
```

![Throughput](../reports/optimization/charts/optimization_throughput.svg)

![Latency](../reports/optimization/charts/optimization_latency.svg)

| step | change | tokens/s | p95 latency (s) | p50 TTFT (s) |
| --- | --- | ---: | ---: | ---: |
| Baseline |  |  |  |  |
| SGLang serving |  |  |  |  |
| 8GB tuned profile |  |  |  |  |
| Concurrency tuned |  |  |  |  |

## Triton Kernels

| kernel | shape | torch ms | triton ms | speedup | max abs error |
| --- | --- | ---: | ---: | ---: | ---: |
| matmul |  |  |  |  |  |
| rmsnorm |  |  |  |  |  |

## Tuning Notes

- Baseline:
- Bottleneck:
- Change made:
- Result:
- Next change:
