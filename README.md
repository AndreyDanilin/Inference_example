# SGLang/Triton Inference Stack Lab

GPU-first pet project for modern LLM inference systems work:

- SGLang OpenAI-compatible serving for `Qwen/Qwen3-0.6B`.
- Streaming benchmark harness with TTFT, latency, throughput, and concurrency summaries.
- Triton kernel lab for transformer-adjacent GPU primitives.
- Hardware profile workflow for tuning the project to a real 8 GB CUDA GPU.

The default path targets an 8 GB NVIDIA GPU. There is no paid provider dependency.

## Stack

- Python + uv
- SGLang for LLM serving
- Triton for custom CUDA kernels
- PyTorch for references and correctness checks
- httpx + pytest for benchmark and test infrastructure

## Hardware Assumptions

Recommended baseline:

- NVIDIA CUDA GPU with at least 8 GB VRAM
- Linux, WSL2, or Docker with NVIDIA Container Toolkit
- Recent NVIDIA driver with CUDA 12.x support
- `Qwen/Qwen3-0.6B` as the default model

`Qwen/Qwen3-1.7B` is included as a stretch config, but tune it after the 0.6B baseline is stable.

## Setup

Install uv, then sync the development dependencies:

```bash
uv python install 3.12
uv sync --extra dev
```

For the CUDA serving and kernel path:

```bash
sudo apt-get update
sudo apt-get install -y libnuma1 nvidia-cuda-toolkit
nvcc --version
uv python install 3.12
uv sync --extra cuda --extra dev
```

Use Python 3.10-3.12 for the CUDA path. The repository includes `.python-version` with `3.12` because SGLang serving dependencies are not a good fit for Python 3.13 yet.

On Windows, use WSL2 or Docker for the CUDA path. The unit tests still run without CUDA and skip GPU checks automatically.

If uv already created a Python 3.13 environment, recreate it:

```bash
rm -rf .venv
uv python install 3.12
uv sync --extra cuda --extra dev
```

## Launch SGLang

```bash
uv run --extra cuda inference-lab serve --config configs/sglang.qwen3-0.6b.yaml
```

Equivalent helper scripts:

```bash
./scripts/serve_sglang.sh
```

```powershell
.\scripts\serve_sglang.ps1
```

The launch config includes 8 GB-oriented defaults:

- `--reasoning-parser qwen3`
- `--mem-fraction-static 0.75`
- `--max-total-tokens 4096`
- `--max-prefill-tokens 2048`
- `--context-length 4096`
- `--enable-metrics`

## Chat Completion Example

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "/no_think Explain continuous batching in one sentence."}
    ],
    "temperature": 0.2,
    "max_tokens": 128
  }'
```

Thinking-mode prompt:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "/think Why does KV cache memory limit concurrency?"}
    ],
    "temperature": 0.2,
    "max_tokens": 128,
    "stream": true
  }'
```

## Benchmark SGLang

Start the server, then run:

```bash
uv run inference-lab bench-sglang \
  --base-url http://127.0.0.1:30000/v1 \
  --model Qwen/Qwen3-0.6B \
  --concurrency 1,2,4,8 \
  --max-tokens 128 \
  --output-dir reports
```

Outputs:

- `reports/sglang_benchmark.json`
- `reports/sglang_benchmark.md`

Example report shape:

| scenario | concurrency | total | ok | errors | p50 latency (s) | p95 latency (s) | p50 TTFT (s) | p95 TTFT (s) | tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1 | 8 | 8 | 0 | 0.720000 | 0.910000 | 0.130000 | 0.180000 | 78.400000 |
| short | 4 | 8 | 8 | 0 | 1.120000 | 1.640000 | 0.190000 | 0.310000 | 156.200000 |
| long | 1 | 8 | 8 | 0 | 1.980000 | 2.410000 | 0.210000 | 0.290000 | 64.900000 |
| long | 4 | 8 | 8 | 0 | 3.360000 | 4.280000 | 0.330000 | 0.520000 | 121.700000 |

The sample numbers above show the report format, not a measurement from this machine.

## Benchmark Triton Kernels

```bash
uv run --extra cuda inference-lab bench-kernels \
  --kernel matmul,rmsnorm \
  --warmup 10 \
  --repeats 50 \
  --output-dir reports
```

The kernel lab currently includes:

- FP16 matmul compared with `torch.matmul`.
- Fused RMSNorm compared with a PyTorch reference.

Example report shape:

| kernel | shape | torch ms | triton ms | speedup | max abs error |
| --- | --- | ---: | ---: | ---: | ---: |
| matmul | 512x512 @ 512x512 fp16 | 0.180000 | 0.150000 | 1.2000x | 0.007812 |
| rmsnorm | 1024x2048 fp16 | 0.090000 | 0.052000 | 1.7308x | 0.003906 |

## Tune To Your Hardware

Capture hardware facts:

```bash
uv run inference-lab hardware-info \
  --profile configs/hardware.8gb.yaml \
  --output reports/hardware.md
```

Then follow [docs/HARDWARE_TUNING.md](docs/HARDWARE_TUNING.md) and write measured results into [docs/RESULTS_TEMPLATE.md](docs/RESULTS_TEMPLATE.md).

The tuning loop is:

1. Capture GPU facts and free VRAM.
2. Run the 0.6B SGLang baseline.
3. Sweep concurrency `1,2,4,8`.
4. Reduce `max_total_tokens`, `max_prefill_tokens`, or `mem_fraction_static` if OOM appears.
5. Run Triton kernel benchmarks.
6. Fill `reports/optimization_steps.yaml` with the major optimization steps.
7. Generate charts for the final report.
8. Commit the final measured markdown report.

Generate optimization charts:

```bash
uv run inference-lab plot-optimization \
  --input docs/sample_reports/optimization_steps.yaml \
  --output-dir reports/optimization
```

This writes:

- `reports/optimization/charts/optimization_throughput.svg`
- `reports/optimization/charts/optimization_latency.svg`
- `reports/optimization/optimization_summary.md`

Use the sample YAML only as the shape of the report. The final GitHub report should replace it with measured numbers from your GPU and show tokens/sec increasing after each major optimization step.

## Docker

```bash
docker build -t sglang-triton-inference-lab .
docker run --gpus all -p 30000:30000 sglang-triton-inference-lab
```

With compose:

```bash
docker compose up --build
```

## Tests

CPU/dev checks:

```bash
uv run --extra dev pytest
```

CUDA checks:

```bash
uv run --extra cuda pytest -m cuda
```

## Project Layout

```text
configs/              SGLang, benchmark, and hardware profiles
docs/                 Tuning notes and result templates
prompts/              Prompt suite for serving benchmarks
scripts/              Thin launch and benchmark wrappers
src/inference_lab/    CLI, configs, reports, SGLang benchmark client, Triton kernels
tests/                Unit tests and CUDA-gated correctness checks
```
