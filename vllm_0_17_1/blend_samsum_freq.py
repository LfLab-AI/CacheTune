"""Run the SAMSum entry point from the migration root."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path[:0] = [str(root / "example"), str(root)]
    runpy.run_path(str(root / "example" / "blend_samsum_freq.py"), run_name="__main__")
