from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class RequestMeasurement:
    scenario: str
    concurrency: int
    latency_s: float
    ttft_s: float | None
    output_tokens: int
    ok: bool
    prompt_name: str | None = None
    input_tokens: int | None = None
    error: str | None = None

    @property
    def tokens_per_s(self) -> float:
        if not self.ok or self.latency_s <= 0:
            return 0.0
        return self.output_tokens / self.latency_s


@dataclass(frozen=True)
class SummaryRow:
    scenario: str
    concurrency: int
    total_requests: int
    successful_requests: int
    error_count: int
    p50_latency_s: float
    p95_latency_s: float
    p50_ttft_s: float | None
    p95_ttft_s: float | None
    mean_tokens_per_s: float
    mean_output_tokens: float


@dataclass(frozen=True)
class BenchmarkReport:
    name: str
    metadata: dict[str, Any]
    measurements: list[RequestMeasurement]
    summaries: list[SummaryRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metadata": self.metadata,
            "measurements": [asdict(measurement) for measurement in self.measurements],
            "summaries": [asdict(summary) for summary in self.summaries],
        }


def aggregate_measurements(measurements: Iterable[RequestMeasurement]) -> list[SummaryRow]:
    groups: dict[tuple[str, int], list[RequestMeasurement]] = {}
    for measurement in measurements:
        groups.setdefault((measurement.scenario, measurement.concurrency), []).append(measurement)

    summaries: list[SummaryRow] = []
    for (scenario, concurrency), rows in sorted(groups.items()):
        successful = [row for row in rows if row.ok]
        latencies = [row.latency_s for row in rows]
        ttfts = [row.ttft_s for row in successful if row.ttft_s is not None]
        token_rates = [row.tokens_per_s for row in successful]
        output_tokens = [row.output_tokens for row in successful]
        summaries.append(
            SummaryRow(
                scenario=scenario,
                concurrency=concurrency,
                total_requests=len(rows),
                successful_requests=len(successful),
                error_count=len(rows) - len(successful),
                p50_latency_s=_round(_percentile_nearest(latencies, 50)),
                p95_latency_s=_round(_percentile_nearest(latencies, 95)),
                p50_ttft_s=_optional_round(_percentile_nearest(ttfts, 50)),
                p95_ttft_s=_optional_round(_percentile_nearest(ttfts, 95)),
                mean_tokens_per_s=_round(mean(token_rates) if token_rates else 0.0),
                mean_output_tokens=_round(mean(output_tokens) if output_tokens else 0.0),
            )
        )
    return summaries


def _percentile_nearest(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    rank = max(1, ceil((percentile / 100) * (len(sorted_values) + 1)))
    return sorted_values[min(rank - 1, len(sorted_values) - 1)]


def _optional_round(value: float | None) -> float | None:
    return None if value is None else _round(value)


def _round(value: float | None) -> float:
    if value is None:
        return 0.0
    return round(float(value), 6)
