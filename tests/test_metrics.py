from inference_lab.metrics import RequestMeasurement, aggregate_measurements


def test_aggregate_measurements_computes_latency_percentiles_and_throughput() -> None:
    measurements = [
        RequestMeasurement(
            scenario="short",
            concurrency=4,
            latency_s=1.0,
            ttft_s=0.20,
            output_tokens=20,
            ok=True,
        ),
        RequestMeasurement(
            scenario="short",
            concurrency=4,
            latency_s=2.0,
            ttft_s=0.30,
            output_tokens=50,
            ok=True,
        ),
        RequestMeasurement(
            scenario="short",
            concurrency=4,
            latency_s=4.0,
            ttft_s=None,
            output_tokens=0,
            ok=False,
            error="timeout",
        ),
    ]

    summary = aggregate_measurements(measurements)

    assert len(summary) == 1
    row = summary[0]
    assert row.scenario == "short"
    assert row.concurrency == 4
    assert row.total_requests == 3
    assert row.successful_requests == 2
    assert row.error_count == 1
    assert row.p50_latency_s == 2.0
    assert row.p95_latency_s == 4.0
    assert row.p50_ttft_s == 0.30
    assert row.mean_tokens_per_s == 22.5
