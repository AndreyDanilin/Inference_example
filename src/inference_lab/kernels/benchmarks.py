from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Iterable


@dataclass(frozen=True)
class KernelCorrectnessResult:
    kernel: str
    passed: bool
    max_abs_error: float


@dataclass(frozen=True)
class KernelBenchmarkResult:
    kernel: str
    shape: str
    torch_ms: float
    triton_ms: float
    speedup: float
    max_abs_error: float


@dataclass(frozen=True)
class KernelBenchmarkReport:
    name: str
    metadata: dict[str, object]
    results: list[KernelBenchmarkResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "metadata": self.metadata,
            "results": [asdict(result) for result in self.results],
        }


def available_cuda() -> bool:
    try:
        import torch
        import triton  # noqa: F401
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def run_kernel_correctness(kernels: Iterable[str]) -> list[KernelCorrectnessResult]:
    _require_available_cuda()
    import torch

    from inference_lab.kernels.ops import torch_rmsnorm, triton_matmul, triton_rmsnorm

    results: list[KernelCorrectnessResult] = []
    for kernel in kernels:
        if kernel == "matmul":
            a = torch.randn((64, 96), device="cuda", dtype=torch.float16)
            b = torch.randn((96, 80), device="cuda", dtype=torch.float16)
            actual = triton_matmul(a, b)
            expected = torch.matmul(a, b)
        elif kernel == "rmsnorm":
            x = torch.randn((32, 256), device="cuda", dtype=torch.float16)
            weight = torch.randn((256,), device="cuda", dtype=torch.float16)
            actual = triton_rmsnorm(x, weight)
            expected = torch_rmsnorm(x, weight)
        else:
            raise ValueError(f"unknown kernel {kernel!r}")
        max_abs_error = float((actual - expected).abs().max().item())
        results.append(
            KernelCorrectnessResult(
                kernel=kernel,
                passed=bool(torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)),
                max_abs_error=max_abs_error,
            )
        )
    return results


def run_kernel_benchmarks(kernels: Iterable[str], warmup: int, repeats: int) -> KernelBenchmarkReport:
    _require_available_cuda()
    import torch

    from inference_lab.kernels.ops import torch_rmsnorm, triton_matmul, triton_rmsnorm

    results: list[KernelBenchmarkResult] = []
    for kernel in kernels:
        if kernel == "matmul":
            a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
            b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
            torch_fn = lambda: torch.matmul(a, b)
            triton_fn = lambda: triton_matmul(a, b)
            expected = torch_fn()
            actual = triton_fn()
            shape = "512x512 @ 512x512 fp16"
        elif kernel == "rmsnorm":
            x = torch.randn((1024, 2048), device="cuda", dtype=torch.float16)
            weight = torch.randn((2048,), device="cuda", dtype=torch.float16)
            torch_fn = lambda: torch_rmsnorm(x, weight)
            triton_fn = lambda: triton_rmsnorm(x, weight)
            expected = torch_fn()
            actual = triton_fn()
            shape = "1024x2048 fp16"
        else:
            raise ValueError(f"unknown kernel {kernel!r}")

        torch_ms = _benchmark_cuda_ms(torch_fn, warmup=warmup, repeats=repeats)
        triton_ms = _benchmark_cuda_ms(triton_fn, warmup=warmup, repeats=repeats)
        max_abs_error = float((actual - expected).abs().max().item())
        results.append(
            KernelBenchmarkResult(
                kernel=kernel,
                shape=shape,
                torch_ms=round(torch_ms, 6),
                triton_ms=round(triton_ms, 6),
                speedup=round(torch_ms / triton_ms, 4) if triton_ms else 0.0,
                max_abs_error=round(max_abs_error, 6),
            )
        )

    return KernelBenchmarkReport(
        name="triton-kernels",
        metadata={"warmup": warmup, "repeats": repeats},
        results=results,
    )


def write_kernel_reports(report: KernelBenchmarkReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "triton_kernel_benchmark.json"
    markdown_path = output_dir / "triton_kernel_benchmark.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_kernel_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _benchmark_cuda_ms(fn: Callable[[], object], *, warmup: int, repeats: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return median(timings)


def _kernel_markdown(report: KernelBenchmarkReport) -> str:
    lines = [
        f"# {report.name} benchmark report",
        "",
        "| kernel | shape | torch ms | triton ms | speedup | max abs error |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in report.results:
        lines.append(
            f"| {result.kernel} | {result.shape} | {result.torch_ms:.6f} | "
            f"{result.triton_ms:.6f} | {result.speedup:.4f}x | {result.max_abs_error:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _require_available_cuda() -> None:
    if not available_cuda():
        raise RuntimeError("CUDA, PyTorch, and Triton are required for kernel benchmarks.")
