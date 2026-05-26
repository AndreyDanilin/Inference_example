from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SGLangHardwareSettings:
    max_total_tokens: int
    max_prefill_tokens: int
    context_length: int
    mem_fraction_static: float
    concurrency: list[int]


@dataclass(frozen=True)
class TritonHardwareSettings:
    matmul_shape: str
    rmsnorm_shape: str
    warmup: int
    repeats: int


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    min_vram_gb: int
    default_model: str
    stretch_model: str | None
    notes: list[str]
    sglang: SGLangHardwareSettings
    triton: TritonHardwareSettings


@dataclass(frozen=True)
class GpuFacts:
    name: str
    driver_version: str
    total_memory_mib: int
    cuda_version: str


def load_hardware_profile(path: Path) -> HardwareProfile:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    sglang = _mapping(payload.get("sglang"), "sglang")
    triton = _mapping(payload.get("triton"), "triton")
    return HardwareProfile(
        name=str(payload["name"]),
        min_vram_gb=int(payload["min_vram_gb"]),
        default_model=str(payload["default_model"]),
        stretch_model=_optional_str(payload.get("stretch_model")),
        notes=[str(note) for note in payload.get("notes", [])],
        sglang=SGLangHardwareSettings(
            max_total_tokens=int(sglang["max_total_tokens"]),
            max_prefill_tokens=int(sglang["max_prefill_tokens"]),
            context_length=int(sglang["context_length"]),
            mem_fraction_static=float(sglang["mem_fraction_static"]),
            concurrency=[int(value) for value in sglang["concurrency"]],
        ),
        triton=TritonHardwareSettings(
            matmul_shape=str(triton["matmul_shape"]),
            rmsnorm_shape=str(triton["rmsnorm_shape"]),
            warmup=int(triton["warmup"]),
            repeats=int(triton["repeats"]),
        ),
    )


def collect_gpu_facts() -> GpuFacts:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,cuda_version",
        "--format=csv",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_nvidia_smi_csv(completed.stdout)


def parse_nvidia_smi_csv(output: str) -> GpuFacts:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("nvidia-smi CSV output must contain a header and one GPU row")
    values = [part.strip() for part in lines[1].split(",")]
    if len(values) < 4:
        raise ValueError("nvidia-smi CSV output must contain name, driver, memory, and CUDA version")
    return GpuFacts(
        name=values[0],
        driver_version=values[1],
        total_memory_mib=_parse_mib(values[2]),
        cuda_version=values[3],
    )


def hardware_markdown(facts: GpuFacts, profile: HardwareProfile) -> str:
    lines = [
        "# Hardware Profile",
        "",
        "## Detected GPU",
        "",
        f"- GPU: `{facts.name}`",
        f"- Driver: `{facts.driver_version}`",
        f"- CUDA: `{facts.cuda_version}`",
        f"- VRAM: `{facts.total_memory_mib} MiB`",
        "",
        "## Selected Profile",
        "",
        f"- Profile: `{profile.name}`",
        f"- Default model: `{profile.default_model}`",
        f"- Stretch model: `{profile.stretch_model or 'none'}`",
        f"- SGLang mem fraction static: `{profile.sglang.mem_fraction_static}`",
        f"- SGLang max total tokens: `{profile.sglang.max_total_tokens}`",
        f"- Benchmark concurrency: `{','.join(str(value) for value in profile.sglang.concurrency)}`",
        f"- Triton matmul shape: `{profile.triton.matmul_shape}`",
        f"- Triton RMSNorm shape: `{profile.triton.rmsnorm_shape}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in profile.notes)
    return "\n".join(lines) + "\n"


def _parse_mib(value: str) -> int:
    number = value.replace("MiB", "").strip()
    return int(number)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
