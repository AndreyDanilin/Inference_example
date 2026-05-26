from __future__ import annotations

import sys

from inference_lab.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["bench-kernels", *sys.argv[1:]]))
