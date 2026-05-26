from pathlib import Path
import tomllib


def test_project_pins_python_below_313_for_cuda_stack() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.10,<3.13"
    assert Path(".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_cuda_extra_installs_sglang_runtime_server_dependencies() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    cuda_dependencies = pyproject["project"]["optional-dependencies"]["cuda"]

    assert "sglang[srt]>=0.5.2" in cuda_dependencies
