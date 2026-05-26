from inference_lab.system_requirements import missing_sglang_runtime_dependencies


def test_linux_sglang_preflight_requires_libnuma() -> None:
    missing = missing_sglang_runtime_dependencies(
        platform_system="Linux",
        library_lookup=lambda name: None,
        executable_lookup=lambda name: "/usr/local/cuda/bin/nvcc",
    )

    assert missing == ["libnuma.so.1"]


def test_linux_sglang_preflight_requires_nvcc() -> None:
    missing = missing_sglang_runtime_dependencies(
        platform_system="Linux",
        library_lookup=lambda name: "libnuma.so.1",
        executable_lookup=lambda name: None,
    )

    assert missing == ["nvcc"]


def test_non_linux_sglang_preflight_skips_native_libraries() -> None:
    missing = missing_sglang_runtime_dependencies(
        platform_system="Windows",
        library_lookup=lambda name: None,
        executable_lookup=lambda name: None,
    )

    assert missing == []
