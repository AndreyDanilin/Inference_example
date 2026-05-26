from __future__ import annotations

import platform
import shutil
from ctypes.util import find_library
from pathlib import Path
from typing import Callable


def missing_sglang_runtime_dependencies(
    *,
    platform_system: str | None = None,
    library_lookup: Callable[[str], str | None] = find_library,
    executable_lookup: Callable[[str], str | None] = shutil.which,
    path_exists: Callable[[str], bool] = lambda path: Path(path).exists(),
) -> list[str]:
    system = platform_system or platform.system()
    if system != "Linux":
        return []

    missing: list[str] = []
    if library_lookup("numa") is None:
        missing.append("libnuma.so.1")
    if executable_lookup("nvcc") is None:
        missing.append("nvcc")
    if executable_lookup("g++") is None:
        missing.append("g++")
    if not path_exists("/usr/include/math.h"):
        missing.append("libc6-dev headers")
    return missing


def format_missing_dependency_help(missing: list[str]) -> str:
    dependencies = ", ".join(missing)
    return (
        f"Missing native runtime dependencies for SGLang: {dependencies}\n"
        "Install them on Ubuntu/WSL with:\n"
        "  sudo apt-get update\n"
        "  sudo apt-get install -y libnuma1 nvidia-cuda-toolkit build-essential libc6-dev\n"
        "Verify with:\n"
        "  nvcc --version\n"
        "  g++ --version\n"
        "  test -f /usr/include/math.h && echo 'math.h OK'\n"
        "Then rerun `uv run --extra cuda inference-lab serve ...`."
    )


missing_sglang_runtime_libraries = missing_sglang_runtime_dependencies
format_missing_library_help = format_missing_dependency_help
