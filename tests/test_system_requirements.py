from inference_lab.system_requirements import missing_sglang_runtime_libraries


def test_linux_sglang_preflight_requires_libnuma() -> None:
    missing = missing_sglang_runtime_libraries(
        platform_system="Linux",
        library_lookup=lambda name: None,
    )

    assert missing == ["libnuma.so.1"]


def test_non_linux_sglang_preflight_skips_native_libraries() -> None:
    missing = missing_sglang_runtime_libraries(
        platform_system="Windows",
        library_lookup=lambda name: None,
    )

    assert missing == []
