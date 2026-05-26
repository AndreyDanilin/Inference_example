from __future__ import annotations

import json
from pathlib import Path

from inference_lab.metrics import BenchmarkReport


def write_json_report(report: BenchmarkReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_markdown_report(report: BenchmarkReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {report.name} benchmark report",
        "",
        "## Metadata",
        "",
    ]
    for key, value in sorted(report.metadata.items()):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| scenario | concurrency | total | ok | errors | p50 latency (s) | p95 latency (s) | p50 TTFT (s) | p95 TTFT (s) | tokens/s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report.summaries:
        lines.append(
            "| "
            f"{row.scenario} | {row.concurrency} | {row.total_requests} | "
            f"{row.successful_requests} | {row.error_count} | "
            f"{row.p50_latency_s:.6f} | {row.p95_latency_s:.6f} | "
            f"{_format_optional(row.p50_ttft_s)} | {_format_optional(row.p95_ttft_s)} | "
            f"{row.mean_tokens_per_s:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"
