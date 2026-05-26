from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from inference_lab.config import load_benchmark_config, load_sglang_config, parse_int_list
from inference_lab.reports import write_json_report, write_markdown_report
from inference_lab.sglang_client import SGLangBenchmarkOptions, run_sglang_benchmark
from inference_lab.system_requirements import (
    format_missing_library_help,
    missing_sglang_runtime_libraries,
)

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inference-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Launch an SGLang OpenAI-compatible server.")
    serve.add_argument("--config", type=Path, default=Path("configs/sglang.qwen3-0.6b.yaml"))
    serve.add_argument("--model", default=None)
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--dry-run", action="store_true")
    serve.set_defaults(func=serve_sglang)

    bench_sglang = subparsers.add_parser("bench-sglang", help="Benchmark an SGLang chat completions endpoint.")
    bench_sglang.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    bench_sglang.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    bench_sglang.add_argument("--model", default="Qwen/Qwen3-0.6B")
    bench_sglang.add_argument("--concurrency", default=None, help="Comma-separated concurrency list.")
    bench_sglang.add_argument("--max-tokens", type=int, default=None)
    bench_sglang.add_argument("--output-dir", type=Path, default=Path("reports"))
    bench_sglang.set_defaults(func=bench_sglang_command)

    bench_kernels = subparsers.add_parser("bench-kernels", help="Benchmark Triton kernels against PyTorch.")
    bench_kernels.add_argument("--kernel", default="matmul,rmsnorm", help="Comma-separated kernel names.")
    bench_kernels.add_argument("--warmup", type=int, default=10)
    bench_kernels.add_argument("--repeats", type=int, default=50)
    bench_kernels.add_argument("--output-dir", type=Path, default=Path("reports"))
    bench_kernels.set_defaults(func=bench_kernels_command)

    hardware = subparsers.add_parser("hardware-info", help="Collect nvidia-smi facts for result reports.")
    hardware.add_argument("--profile", type=Path, default=Path("configs/hardware.8gb.yaml"))
    hardware.add_argument("--output", type=Path, default=Path("reports/hardware.md"))
    hardware.set_defaults(func=hardware_info_command)

    optimize = subparsers.add_parser("plot-optimization", help="Generate optimization progression charts.")
    optimize.add_argument("--input", type=Path, default=Path("docs/sample_reports/optimization_steps.yaml"))
    optimize.add_argument("--output-dir", type=Path, default=Path("reports/optimization"))
    optimize.set_defaults(func=plot_optimization_command)

    return parser


def serve_sglang(args: argparse.Namespace) -> int:
    missing_libraries = missing_sglang_runtime_libraries()
    if missing_libraries:
        console.print(f"[red]{format_missing_library_help(missing_libraries)}[/red]")
        return 2

    config = load_sglang_config(args.config)
    if args.model:
        config = _replace_dataclass(config, model_path=args.model)
    if args.host:
        config = _replace_dataclass(config, host=args.host)
    if args.port:
        config = _replace_dataclass(config, port=args.port)

    command = [sys.executable, "-m", "sglang.launch_server", *config.to_launch_args()]
    console.print("[bold]Launching SGLang[/bold]")
    console.print(" ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def bench_sglang_command(args: argparse.Namespace) -> int:
    config = load_benchmark_config(args.config)
    concurrency = parse_int_list(args.concurrency) if args.concurrency else config.concurrency
    max_tokens = args.max_tokens if args.max_tokens is not None else config.max_tokens
    options = SGLangBenchmarkOptions(
        base_url=args.base_url,
        model=args.model,
        prompt_file=config.prompt_file,
        concurrency=concurrency,
        max_tokens=max_tokens,
        requests_per_concurrency=config.requests_per_concurrency,
        warmup_requests=config.warmup_requests,
        timeout_s=config.timeout_s,
        temperature=config.temperature,
        stream=config.stream,
    )
    report = asyncio.run(run_sglang_benchmark(options))
    json_path = write_json_report(report, args.output_dir / "sglang_benchmark.json")
    markdown_path = write_markdown_report(report, args.output_dir / "sglang_benchmark.md")
    console.print(f"Wrote {json_path}")
    console.print(f"Wrote {markdown_path}")
    return 0


def bench_kernels_command(args: argparse.Namespace) -> int:
    from inference_lab.kernels.benchmarks import run_kernel_benchmarks, write_kernel_reports

    kernels = [kernel.strip() for kernel in args.kernel.split(",") if kernel.strip()]
    report = run_kernel_benchmarks(kernels=kernels, warmup=args.warmup, repeats=args.repeats)
    json_path, markdown_path = write_kernel_reports(report, args.output_dir)
    console.print(f"Wrote {json_path}")
    console.print(f"Wrote {markdown_path}")
    return 0


def hardware_info_command(args: argparse.Namespace) -> int:
    from inference_lab.hardware import collect_gpu_facts, hardware_markdown, load_hardware_profile

    profile = load_hardware_profile(args.profile)
    facts = collect_gpu_facts()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(hardware_markdown(facts, profile), encoding="utf-8")
    console.print(f"Wrote {args.output}")
    return 0


def plot_optimization_command(args: argparse.Namespace) -> int:
    from inference_lab.optimization_report import (
        load_optimization_steps,
        validate_speed_progression,
        write_optimization_charts,
        write_optimization_markdown,
    )

    steps = load_optimization_steps(args.input)
    chart_paths = write_optimization_charts(steps, args.output_dir)
    markdown_path = write_optimization_markdown(
        steps,
        args.output_dir / "optimization_summary.md",
        chart_paths,
    )
    console.print(f"Wrote {chart_paths['throughput']}")
    console.print(f"Wrote {chart_paths['latency']}")
    console.print(f"Wrote {markdown_path}")
    if not validate_speed_progression(steps):
        console.print("[yellow]Warning:[/yellow] tokens/s is not strictly increasing across steps.")
    return 0


def _replace_dataclass(instance, **changes):
    from dataclasses import replace

    return replace(instance, **changes)
