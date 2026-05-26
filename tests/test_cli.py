from argparse import Namespace

from inference_lab import cli


def test_serve_exits_with_help_when_native_libraries_are_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "missing_sglang_runtime_libraries",
        lambda: ["libnuma.so.1"],
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
    assert "sudo apt-get update && sudo apt-get install -y libnuma1" in captured.out
