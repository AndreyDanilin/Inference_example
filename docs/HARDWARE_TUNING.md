# Hardware Tuning Notes

This project is intentionally GPU-first. The default profile targets an 8 GB CUDA GPU with `Qwen/Qwen3-0.6B`.

## Capture The Machine

```bash
uv run inference-lab hardware-info --profile configs/hardware.8gb.yaml --output reports/hardware.md
nvidia-smi
```

Record the GPU name, driver, CUDA version, total VRAM, and free VRAM before starting SGLang.

## First Knobs For 8 GB VRAM

- Keep `Qwen/Qwen3-0.6B` as the default model until the baseline benchmark is stable.
- Start with `mem_fraction_static: 0.75`, `max_total_tokens: 4096`, `max_prefill_tokens: 2048`, and `context_length: 4096`.
- If SGLang reports OOM, reduce `max_total_tokens` first, then `max_prefill_tokens`, then `mem_fraction_static`.
- If benchmark concurrency 8 is unstable, run `--concurrency 1,2,4` and add 8 only after checking free VRAM.
- Treat `Qwen/Qwen3-1.7B` as a stretch profile; use it only after the 0.6B report is complete.

## Result Workflow

1. Capture hardware facts into `reports/hardware.md`.
2. Start SGLang with the 8 GB profile.
3. Run `bench-sglang` with `--concurrency 1,2,4,8`.
4. Run `bench-kernels` for `matmul,rmsnorm`.
5. Record major optimization steps in `reports/optimization_steps.yaml`.
6. Generate optimization charts with `uv run inference-lab plot-optimization --input reports/optimization_steps.yaml --output-dir reports/optimization`.
7. Copy the measured tables and charts into `docs/RESULTS.md` or use `docs/RESULTS_TEMPLATE.md`.

The final report should make the optimization story visible:

- Baseline before tuning.
- SGLang serving baseline.
- 8 GB VRAM profile after token/memory tuning.
- Best stable concurrency from the sweep.
- Optional stretch model or extra kernel work if it improves the measured result.
