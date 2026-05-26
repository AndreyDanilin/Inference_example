import json
from pathlib import Path

from inference_lab.metrics import BenchmarkReport, RequestMeasurement, aggregate_measurements
from inference_lab.reports import write_json_report, write_markdown_report


def test_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    measurements = [
        RequestMeasurement(
            scenario="short",
            concurrency=1,
            latency_s=1.5,
            ttft_s=0.25,
            output_tokens=30,
            ok=True,
        )
    ]
    report = BenchmarkReport(
        name="sglang",
        metadata={"model": "Qwen/Qwen3-0.6B"},
        measurements=measurements,
        summaries=aggregate_measurements(measurements),
    )

    json_path = write_json_report(report, tmp_path / "report.json")
    md_path = write_markdown_report(report, tmp_path / "report.md")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")

    assert payload["name"] == "sglang"
    assert payload["metadata"]["model"] == "Qwen/Qwen3-0.6B"
    assert "| short | 1 | 1 | 1 | 0 |" in markdown
