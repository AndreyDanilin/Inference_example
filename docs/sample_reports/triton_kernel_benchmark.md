# triton-kernels benchmark report

This is an example report shape for README/docs. Replace it with measured output from `uv run inference-lab bench-kernels`.

| kernel | shape | torch ms | triton ms | speedup | max abs error |
| --- | --- | ---: | ---: | ---: | ---: |
| matmul | 512x512 @ 512x512 fp16 | 0.180000 | 0.150000 | 1.2000x | 0.007812 |
| rmsnorm | 1024x2048 fp16 | 0.090000 | 0.052000 | 1.7308x | 0.003906 |
