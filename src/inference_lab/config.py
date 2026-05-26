from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SGLangConfig:
    model_path: str
    host: str = "0.0.0.0"
    port: int = 30000
    reasoning_parser: str | None = "qwen3"
    log_level: str = "info"
    mem_fraction_static: float | None = 0.75
    max_total_tokens: int | None = None
    max_prefill_tokens: int | None = None
    context_length: int | None = None
    enable_metrics: bool = True
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def to_launch_args(self) -> list[str]:
        args = [
            "--model-path",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--log-level",
            self.log_level,
        ]
        if self.reasoning_parser:
            args.extend(["--reasoning-parser", self.reasoning_parser])
        if self.enable_metrics:
            args.append("--enable-metrics")
        if self.mem_fraction_static is not None:
            args.extend(["--mem-fraction-static", str(self.mem_fraction_static)])
        if self.max_total_tokens is not None:
            args.extend(["--max-total-tokens", str(self.max_total_tokens)])
        if self.max_prefill_tokens is not None:
            args.extend(["--max-prefill-tokens", str(self.max_prefill_tokens)])
        if self.context_length is not None:
            args.extend(["--context-length", str(self.context_length)])
        args.extend(self.extra_args)
        return args


@dataclass(frozen=True)
class BenchmarkConfig:
    concurrency: list[int]
    max_tokens: int
    warmup_requests: int
    requests_per_concurrency: int
    prompt_file: Path
    timeout_s: float = 120.0
    temperature: float = 0.2
    stream: bool = True


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def load_sglang_config(path: Path) -> SGLangConfig:
    payload = load_yaml(path)
    server = payload.get("server", payload)
    if not isinstance(server, dict):
        raise ValueError(f"{path} server section must be a mapping")

    model_path = server.get("model_path") or server.get("model")
    if not model_path:
        raise ValueError(f"{path} must define server.model_path")

    extra_args = server.get("extra_args", ())
    if extra_args is None:
        extra_args = ()
    if not isinstance(extra_args, list | tuple):
        raise ValueError("server.extra_args must be a list")

    return SGLangConfig(
        model_path=str(model_path),
        host=str(server.get("host", "0.0.0.0")),
        port=int(server.get("port", 30000)),
        reasoning_parser=_optional_str(server.get("reasoning_parser", "qwen3")),
        log_level=str(server.get("log_level", "info")),
        mem_fraction_static=_optional_float(server.get("mem_fraction_static", 0.75)),
        max_total_tokens=_optional_int(server.get("max_total_tokens")),
        max_prefill_tokens=_optional_int(server.get("max_prefill_tokens")),
        context_length=_optional_int(server.get("context_length")),
        enable_metrics=bool(server.get("enable_metrics", True)),
        extra_args=tuple(str(arg) for arg in extra_args),
    )


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    payload = load_yaml(path)
    benchmark = payload.get("benchmark", payload)
    if not isinstance(benchmark, dict):
        raise ValueError(f"{path} benchmark section must be a mapping")

    concurrency = benchmark.get("concurrency", [1])
    if isinstance(concurrency, str):
        concurrency_values = parse_int_list(concurrency)
    elif isinstance(concurrency, list):
        concurrency_values = [int(value) for value in concurrency]
    else:
        raise ValueError("benchmark.concurrency must be a list or comma-separated string")

    return BenchmarkConfig(
        concurrency=concurrency_values,
        max_tokens=int(benchmark.get("max_tokens", 128)),
        warmup_requests=int(benchmark.get("warmup_requests", 2)),
        requests_per_concurrency=int(benchmark.get("requests_per_concurrency", 8)),
        prompt_file=Path(str(benchmark.get("prompt_file", "prompts/chat_prompts.yaml"))),
        timeout_s=float(benchmark.get("timeout_s", 120.0)),
        temperature=float(benchmark.get("temperature", 0.2)),
        stream=bool(benchmark.get("stream", True)),
    )


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
