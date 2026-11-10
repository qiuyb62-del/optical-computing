"""Measure fixed static-error 4x4 SVD circuits against the Python snapshot.

Each condition samples one device once (same seed) and writes its actual phases
into one Lumerical circuit.  H is extracted only after the device is fixed.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumerical.experiments import measure_svd_nonideal  # noqa: E402
from lumerical.interconnect import save_json, timestamp, app_version, absolute_error, relative_error  # noqa: E402
from src.nonideal import NonidealConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "progress/07_lumerical/nonideal4")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--device-seed", type=int, default=20261110)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    W = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    conditions = [
        ("phase_0.01", NonidealConfig(phase_std_rad=0.01)),
        ("loss_0.1db", NonidealConfig(loss_db_per_mzi=0.1)),
        ("bits_6", NonidealConfig(phase_bits=6)),
        ("coupler_0.49", NonidealConfig(coupler_through=0.49)),
        ("combined", NonidealConfig(phase_std_rad=0.01, loss_db_per_mzi=0.1,
                                    phase_bits=8, coupler_through=0.49)),
    ]
    cases = []
    for name, config in conditions:
        case = measure_svd_nonideal(W, args.out, name, config, seed=args.device_seed)
        cases.append(case)
        print(f"{name}: {'PASS' if case['passed'] else 'FAIL'} "
              f"relative={case['relative_error']:.3e}", flush=True)
    payload = {
        "stage": "lumerical_svd4_nonideal",
        "timestamp": timestamp(),
        "lumerical": app_version(),
        "seed": args.seed,
        "device_seed": args.device_seed,
        "tol": 1e-6,
        "W": {"real": W.real.tolist(), "imag": W.imag.tolist()},
        "cases": cases,
        "passed": all(c["passed"] and c.get("restored_passed", True) for c in cases),
        "complete": all(c.get("complete", False) for c in cases),
        "note": "One fixed device per condition; actual phases and loss written from the Python RealizedMesh snapshot.",
    }
    save_json(args.out / "nonideal4.json", payload)
    print(f"nonideal4 overall: {'PASS' if payload['passed'] else 'FAIL'}", flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
