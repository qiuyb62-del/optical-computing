"""Build and measure the complete 4x4 SVD optical circuit in INTERCONNECT.

Every case uses the repository ``compile_svd``; no dense target matrix is
inserted into the circuit.  The measured forward matrix is compared with W/s
and, after the digital recovery s, with W.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumerical.experiments import measure_svd, write_svd_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "progress/07_lumerical/svd4")
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    random_w = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rank_one = np.outer(np.array([1.0, -0.5j, 0.25, 2.0]),
                        np.array([0.3, -1.0, 0.7j, 0.4]))
    cases = [
        ("random_complex", random_w),
        ("identity", np.eye(4, dtype=complex)),
        ("rank_one", rank_one),
        ("zero", np.zeros((4, 4), dtype=complex)),
    ]
    results = []
    for name, W in cases:
        case = measure_svd(W, args.out, name)
        results.append(case)
        if name == "zero":
            print(f"zero: {'PASS' if case['passed'] else 'FAIL'} "
                  f"power_leakage max={case['max_zero_power_w']:.3e} W "
                  f"amplitude_bound={case['amplitude_upper_bound']:.3e}")
            continue
        restored_error = case.get("restored_relative_error")
        restored_abs = case.get("restored_absolute_error")
        print(f"{name}: {'PASS' if case['passed'] else 'FAIL'} "
              f"H-vs-W/s={case['relative_error'] if case['relative_error'] is not None else case['absolute_error']:.3e} "
              f"restored={restored_error if restored_error is not None else restored_abs:.3e} "
              f"complete={case.get('complete')}")
    report = write_svd_report(args.out, results, seed=args.seed)
    print(f"SVD4 overall: {'PASS' if report['passed'] else 'FAIL'}; "
          f"complete={report.get('complete')} -> {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
