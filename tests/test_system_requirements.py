from inference_lab.system_requirements import missing_sglang_runtime_dependencies


def test_linux_sglang_preflight_requires_libnuma() -> None:
    missing = missing_sglang_runtime_dependencies(
        platform_system="Linux",
        library_lookup=lambda name: None,
        executable_lookup=lambda name: "/usr/local/cuda/bin/nvcc",
        path_exists=lambda path: True,
    )

    assert missing == ["libnuma.so.1"]


def test_linux_sglang_preflight_requires_nvcc() -> None:
    missing = missing_sglang_runtime_dependencies(
        platform_system="Linux",
        library_lookup=lambda name: "libnuma.so.1",
        executable_lookup=lambda name: None if name == "nvcc" else f"/usr/bin/{name}",
        path_exists=lambda path: True,
    )

    assert missing == ["nvcc"]


def test_linux_sglang_preflight_requires_build_headers_for_flashinfer_jit() -> None:
    missing = missing_sglang_runtime_dependencies(
        platform_system="Linux",
        library_lookup=lambda name: "libnuma.so.1",
        executable_lookup=lambda name: f"/usr/bin/{name}",
        path_exists=lambda path: False if path == "/usr/include/math.h" else True,
    )

    assert missing == ["libc6-dev headers"]


def test_linux_sglang_preflight_requires_gpp_for_cuda_extensions() -> None:
    missing = missing_sglang_runtime_dependencies(
        platform_system="Linux",
        library_lookup=lambda name: "libnuma.so.1",
        executable_lookup=lambda name: None if name == "g++" else f"/usr/bin/{name}",
        path_exists=lambda path: True,
    )

    assert missing == ["g++"]


def test_non_linux_sglang_preflight_skips_native_libraries() -> None:
    missing = missing_sglang_runtime_dependencies(
        platform_system="Windows",
        library_lookup=lambda name: None,
        executable_lookup=lambda name: None,
        path_exists=lambda path: False,
    )

    assert missing == []
