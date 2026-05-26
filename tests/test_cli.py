from argparse import Namespace

from inference_lab import cli


def test_serve_exits_with_help_when_native_libraries_are_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "missing_sglang_runtime_dependencies",
        lambda: ["libnuma.so.1", "nvcc", "g++", "libc6-dev headers"],
    )

    exit_code = cli.serve_sglang(
        Namespace(
            config=None,
            model=None,
            host=None,
            port=None,
            dry_run=True,
        )
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "sudo apt-get install -y" in captured.out
    assert "nvidia-cuda-toolkit" in captured.out
    assert "build-essential" in captured.out
    assert "libc6-dev" in captured.out
    assert "nvcc --version" in captured.out
    assert "test -f /usr/include/math.h" in captured.out
