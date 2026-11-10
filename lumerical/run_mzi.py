"""Run and record the Lumerical single-MZI complex-matrix validation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumerical.experiments import measure_mzi  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "progress/07_lumerical/single_mzi")
    args = parser.parse_args()
    payload = measure_mzi(args.out)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
