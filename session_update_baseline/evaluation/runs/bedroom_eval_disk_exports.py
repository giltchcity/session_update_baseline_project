"""Run the installed evaluator with the large per-visit exports on the result volume."""
import runpy
import sys
import tempfile
from pathlib import Path


def main():
    scratch = Path(sys.argv.pop(1)).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    sys.argv.pop(0)
    evaluator = Path(sys.argv[0]).resolve(strict=True)
    sys.path.insert(0, str(evaluator.parent))
    make_temp = tempfile.mkdtemp

    def make_workdir(suffix=None, prefix=None, dir=None):
        if prefix == "eval-fast-exports-":
            dir = scratch
        return make_temp(suffix=suffix, prefix=prefix, dir=dir)

    tempfile.mkdtemp = make_workdir
    try:
        runpy.run_path(str(evaluator), run_name="__main__")
    finally:
        tempfile.mkdtemp = make_temp


if __name__ == "__main__":
    main()
