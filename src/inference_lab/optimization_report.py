from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class OptimizationStep:
    name: str
    description: str
    tokens_per_second: float
    p95_latency_s: float
    p50_ttft_s: float | None = None
    notes: str | None = None


def load_optimization_steps(path: Path) -> list[OptimizationStep]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"{path} must contain a non-empty steps list")

    steps: list[OptimizationStep] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ValueError(f"step #{index} must be a mapping")
        steps.append(
            OptimizationStep(
                name=str(raw_step["name"]),
                description=str(raw_step.get("description", "")),
                tokens_per_second=float(raw_step["tokens_per_second"]),
                p95_latency_s=float(raw_step["p95_latency_s"]),
                p50_ttft_s=_optional_float(raw_step.get("p50_ttft_s")),
                notes=_optional_str(raw_step.get("notes")),
            )
        )
    return steps


def validate_speed_progression(steps: Iterable[OptimizationStep]) -> bool:
    values = [step.tokens_per_second for step in steps]
    return all(current > previous for previous, current in zip(values, values[1:]))


def write_optimization_charts(steps: list[OptimizationStep], output_dir: Path) -> dict[str, Path]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    throughput_path = chart_dir / "optimization_throughput.svg"
    latency_path = chart_dir / "optimization_latency.svg"

    throughput_path.write_text(
        _line_chart_svg(
            steps,
            title="Inference throughput by optimization step",
            y_label="tokens/s",
            value_getter=lambda step: step.tokens_per_second,
            higher_is_better=True,
        ),
        encoding="utf-8",
    )
    latency_path.write_text(
        _line_chart_svg(
            steps,
            title="p95 latency by optimization step",
            y_label="p95 latency (s)",
            value_getter=lambda step: step.p95_latency_s,
            higher_is_better=False,
        ),
        encoding="utf-8",
    )
    return {"throughput": throughput_path, "latency": latency_path}


def write_optimization_markdown(
    steps: list[OptimizationStep],
    path: Path,
    chart_paths: dict[str, Path],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    throughput_rel = _relative_markdown_path(path, chart_paths["throughput"])
    latency_rel = _relative_markdown_path(path, chart_paths["latency"])
    lines = [
        "# Optimization Progression",
        "",
        "These charts track the measured speed impact of each major inference optimization step.",
        "",
        f"![Throughput]({throughput_rel})",
        "",
        f"![Latency]({latency_rel})",
        "",
        "| step | tokens/s | p95 latency (s) | p50 TTFT (s) | description |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for step in steps:
        lines.append(
            f"| {step.name} | {step.tokens_per_second:.3f} | {step.p95_latency_s:.3f} | "
            f"{_format_optional(step.p50_ttft_s)} | {step.description} |"
        )
    lines.extend(
        [
            "",
            f"- Speed progression monotonic: `{validate_speed_progression(steps)}`",
            "- Replace the sample YAML values with measured benchmark output before publishing final results.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _line_chart_svg(
    steps: list[OptimizationStep],
    *,
    title: str,
    y_label: str,
    value_getter,
    higher_is_better: bool,
) -> str:
    width = 960
    height = 420
    left = 72
    right = 32
    top = 56
    bottom = 96
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(value_getter(step)) for step in steps]
    max_value = max(values)
    min_value = min(values)
    span = max(max_value - min_value, max_value * 0.1, 1.0)
    y_min = max(0.0, min_value - span * 0.12)
    y_max = max_value + span * 0.16
    y_span = max(y_max - y_min, 1.0)

    def point(index: int, value: float) -> tuple[float, float]:
        if len(steps) == 1:
            x = left + plot_width / 2
        else:
            x = left + (plot_width * index / (len(steps) - 1))
        y = top + plot_height - ((value - y_min) / y_span * plot_height)
        return x, y

    points = [point(index, value) for index, value in enumerate(values)]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    color = "#0f766e" if higher_is_better else "#b45309"
    grid_lines = []
    for tick in range(5):
        y = top + plot_height * tick / 4
        value = y_max - y_span * tick / 4
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb" />'
        )
        grid_lines.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#4b5563">{value:.1f}</text>'
        )

    labels = []
    circles = []
    for index, (step, value, (x, y)) in enumerate(zip(steps, values, points)):
        labels.append(
            f'<text x="{x:.2f}" y="{height - 58}" text-anchor="middle" font-size="12" fill="#111827">{escape(step.name)}</text>'
        )
        labels.append(
            f'<text x="{x:.2f}" y="{height - 40}" text-anchor="middle" font-size="11" fill="#4b5563">{index + 1}</text>'
        )
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" />'
            f'<text x="{x:.2f}" y="{y - 12:.2f}" text-anchor="middle" font-size="12" fill="#111827">{value:.2f}</text>'
        )

    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420" role="img">',
            f"<title>{escape(title)}</title>",
            '<rect width="960" height="420" fill="#ffffff" />',
            f'<text x="{left}" y="30" font-size="22" font-weight="700" fill="#111827">{escape(title)}</text>',
            f'<text x="{left}" y="50" font-size="13" fill="#4b5563">{escape(y_label)}</text>',
            *grid_lines,
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#9ca3af" />',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#9ca3af" />',
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3" />',
            *circles,
            *labels,
            "</svg>",
        ]
    )


def _relative_markdown_path(markdown_path: Path, asset_path: Path) -> str:
    return asset_path.relative_to(markdown_path.parent).as_posix()


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
