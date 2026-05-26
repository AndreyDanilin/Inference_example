from pathlib import Path

from inference_lab.hardware import load_hardware_profile, parse_nvidia_smi_csv


def test_load_hardware_profile_for_8gb_cuda_target() -> None:
    profile = load_hardware_profile(Path("configs/hardware.8gb.yaml"))

    assert profile.name == "8gb-cuda"
    assert profile.min_vram_gb == 8
    assert profile.default_model == "Qwen/Qwen3-0.6B"
    assert profile.sglang.max_total_tokens == 4096
    assert profile.sglang.mem_fraction_static == 0.75


def test_parse_nvidia_smi_csv_extracts_gpu_facts() -> None:
    output = """name, driver_version, memory.total [MiB], cuda_version
NVIDIA GeForce RTX 4060 Laptop GPU, 552.22, 8188 MiB, 12.4
"""

    facts = parse_nvidia_smi_csv(output)

    assert facts.name == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert facts.driver_version == "552.22"
    assert facts.total_memory_mib == 8188
    assert facts.cuda_version == "12.4"
