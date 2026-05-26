from pathlib import Path

from inference_lab.optimization_report import (
    load_optimization_steps,
    validate_speed_progression,
    write_optimization_charts,
    write_optimization_markdown,
)


def test_load_optimization_steps_and_validate_speed_progression() -> None:
    steps = load_optimization_steps(Path("docs/sample_reports/optimization_steps.yaml"))

    assert [step.name for step in steps] == [
        "Baseline",
        "SGLang serving",
        "8GB tuned profile",
        "Concurrency tuned",
    ]
    assert validate_speed_progression(steps)
    assert steps[-1].tokens_per_second > steps[0].tokens_per_second


def test_writes_optimization_svg_charts_and_markdown(tmp_path: Path) -> None:
    steps = load_optimization_steps(Path("docs/sample_reports/optimization_steps.yaml"))

    chart_paths = write_optimization_charts(steps, tmp_path)
    markdown_path = write_optimization_markdown(steps, tmp_path / "optimization_summary.md", chart_paths)

    throughput_svg = chart_paths["throughput"].read_text(encoding="utf-8")
    latency_svg = chart_paths["latency"].read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")

    assert "<svg" in throughput_svg
    assert "tokens/s" in throughput_svg
    assert "p95 latency" in latency_svg
    assert "![Throughput](charts/optimization_throughput.svg)" in markdown
    assert "| Concurrency tuned |" in markdown
