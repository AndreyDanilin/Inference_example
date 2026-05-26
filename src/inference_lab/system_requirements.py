from __future__ import annotations

import platform
from ctypes.util import find_library
from typing import Callable


def missing_sglang_runtime_libraries(
    *,
    platform_system: str | None = None,
    library_lookup: Callable[[str], str | None] = find_library,
) -> list[str]:
    system = platform_system or platform.system()
    if system != "Linux":
        return []

    missing: list[str] = []
    if library_lookup("numa") is None:
        missing.append("libnuma.so.1")
    return missing


def format_missing_library_help(missing: list[str]) -> str:
    libraries = ", ".join(missing)
    return (
        f"Missing native runtime libraries for SGLang: {libraries}\n"
        "Install them on Ubuntu/WSL with:\n"
        "  sudo apt-get update && sudo apt-get install -y libnuma1\n"
        "Then rerun `uv run --extra cuda inference-lab serve ...`."
    )
