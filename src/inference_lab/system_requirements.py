from __future__ import annotations

import platform
import shutil
from ctypes.util import find_library
from typing import Callable


def missing_sglang_runtime_dependencies(
    *,
    platform_system: str | None = None,
    library_lookup: Callable[[str], str | None] = find_library,
    executable_lookup: Callable[[str], str | None] = shutil.which,
) -> list[str]:
    system = platform_system or platform.system()
    if system != "Linux":
        return []

    missing: list[str] = []
    if library_lookup("numa") is None:
        missing.append("libnuma.so.1")
    if executable_lookup("nvcc") is None:
        missing.append("nvcc")
    return missing


def format_missing_dependency_help(missing: list[str]) -> str:
    dependencies = ", ".join(missing)
    return (
        f"Missing native runtime dependencies for SGLang: {dependencies}\n"
        "Install them on Ubuntu/WSL with:\n"
        "  sudo apt-get update\n"
        "  sudo apt-get install -y libnuma1 nvidia-cuda-toolkit\n"
        "Verify with:\n"
        "  nvcc --version\n"
        "Then rerun `uv run --extra cuda inference-lab serve ...`."
    )


missing_sglang_runtime_libraries = missing_sglang_runtime_dependencies
format_missing_library_help = format_missing_dependency_help
