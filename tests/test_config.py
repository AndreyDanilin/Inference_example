from pathlib import Path

from inference_lab.config import load_benchmark_config, load_sglang_config


def test_loads_default_sglang_profile() -> None:
    config = load_sglang_config(Path("configs/sglang.qwen3-0.6b.yaml"))

    assert config.model_path == "Qwen/Qwen3-0.6B"
    assert config.host == "0.0.0.0"
    assert config.port == 30000
    assert config.reasoning_parser == "qwen3"
    assert "--enable-metrics" in config.to_launch_args()
    assert "--mem-fraction-static" in config.to_launch_args()
    assert "--max-total-tokens" in config.to_launch_args()
    assert "--max-prefill-tokens" in config.to_launch_args()
    assert "--context-length" in config.to_launch_args()


def test_loads_benchmark_concurrency_profile() -> None:
    config = load_benchmark_config(Path("configs/benchmark.yaml"))

    assert config.concurrency == [1, 4, 8]
    assert config.max_tokens == 128
    assert config.warmup_requests == 2
    assert config.requests_per_concurrency == 8
