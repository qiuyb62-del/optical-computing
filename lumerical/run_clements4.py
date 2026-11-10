"""Build and measure 4-channel Clements meshes in INTERCONNECT.

Cases are fixed before running: a QR random unitary, identity, reversal
permutation and a diagonal phase matrix.  The same Python decomposition is used
for every case; no phase is adjusted to match a measured answer.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumerical.experiments import measure_unitary, write_clements_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "progress/07_lumerical/clements4")
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, r = np.linalg.qr(z)
    random_unitary = q * (np.diag(r) / np.abs(np.diag(r)))[None, :]
    cases = [
        ("random_unitary", random_unitary),
        ("identity", np.eye(4, dtype=complex)),
        ("reversal_permutation", np.eye(4, dtype=complex)[::-1]),
        ("diagonal_phases", np.diag(np.exp(1j * np.array([0.2, 1.1, -2.0, 2.7])))),
    ]
    results = []
    for name, U in cases:
        case = measure_unitary(U, args.out, name)
        results.append(case)
        print(f"{name}: {'PASS' if case['passed'] else 'FAIL'} "
              f"relative={case['relative_error']:.3e}")
    report = write_clements_report(args.out, results, seed=args.seed)
    print(f"Clements4 overall: {'PASS' if report['passed'] else 'FAIL'}; "
          f"complete={report.get('complete')} -> {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
