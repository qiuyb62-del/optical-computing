"""End-to-end time-domain 4x4 SVD test with coherent I/Q readout.

The same W is first measured as a frequency-domain complex matrix, then a
separate time-domain circuit is driven by a fixed sequence containing positive,
negative, complex and zero components.  I/Q is obtained from 16 photocurrents;
there is no per-sample phase fitting.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumerical.experiments import measure_svd  # noqa: E402
from lumerical.interconnect import save_json, timestamp, app_version  # noqa: E402
from lumerical.time_domain import measure_svd_time_domain  # noqa: E402
from lumerical.validation import compare_complex, time_domain_passed  # noqa: E402
from src.svd_matmul import compile_svd  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "progress/07_lumerical/svd4_td")
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    W = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    circuit = compile_svd(W)
    print("frequency-domain reference", flush=True)
    freq_case = measure_svd(W, args.out, "frequency_reference")
    H = np.asarray(freq_case["H"]["real"]) + 1j * np.asarray(freq_case["H"]["imag"])
    s = float(freq_case["recovery_scale"])

    vectors = [
        np.array([0.8, -0.5, 0.3 + 0.4j, -0.2j], dtype=complex),
        np.array([1.0j, -1.0, 0.5, 0.0], dtype=complex),
        np.array([2.0, -1.0, 0.5j, 0.0], dtype=complex),
    ]
    print("time-domain sequence", flush=True)
    td = measure_svd_time_domain(W, args.out, vectors, name="svd4_time_domain")
    records = []
    for record in td["records"]:
        x = np.asarray(record["x"]["real"]) + 1j * np.asarray(record["x"]["imag"])
        y = np.asarray(record["measured_optical_output"]["real"]) + \
            1j * np.asarray(record["measured_optical_output"]["imag"])
        g = record["encoding_scale_g"]
        expected_optical = H @ (x / g)
        expected_restored = W @ x
        measured_restored = s * g * y
        optical_scale = float(np.linalg.norm(H) * np.linalg.norm(x / g))
        restored_scale = float(np.linalg.norm(W) * np.linalg.norm(x))
        optical_check = compare_complex(y, expected_optical, scale=optical_scale)
        restored_check = compare_complex(measured_restored, expected_restored,
                                         scale=restored_scale)
        checks = {"optical_Hx": optical_check, "restored_Wx": restored_check}
        records.append({
            "index": record["index"],
            "x": record["x"],
            "encoding_scale_g": g,
            "measured_optical_output": record["measured_optical_output"],
            "expected_optical_Hx": {"real": expected_optical.real.tolist(),
                                    "imag": expected_optical.imag.tolist()},
            "expected_restored_Wx": {"real": expected_restored.real.tolist(),
                                     "imag": expected_restored.imag.tolist()},
            "optical_relative_error": optical_check["relative_error"],
            "optical_absolute_error": optical_check["absolute_error"],
            "restored_output_sg_y": {"real": measured_restored.real.tolist(),
                                     "imag": measured_restored.imag.tolist()},
            "restored_relative_error": restored_check["relative_error"],
            "restored_absolute_error": restored_check["absolute_error"],
            "checks": checks,
            "passed": bool(optical_check["passed"] and restored_check["passed"]),
        })
        print(f"vector {record['index']}: optical rel="
              f"{records[-1]['optical_relative_error']} restored rel="
              f"{records[-1]['restored_relative_error']}", flush=True)
    payload = {
        "stage": "lumerical_svd4_time_domain",
        "timestamp": timestamp(),
        "lumerical": app_version(),
        "seed": args.seed,
        "label": "coherent I/Q time-domain validation, not intensity detection",
        "frequency_matrix_relative_error": freq_case["relative_error"],
        "recovery_scale": s,
        "input_encoding": "x/g at four amplitude/phase encoders; g=max(1,max|x|)",
        "receiver": "one common CW; 0/pi2 LO hybrid; balanced subtraction of PD powers",
        "receiver_details": {
            "cw_power_w": 1.0,
            "cw_frequency_hz": 193.1e12,
            "cw_linewidth_hz": 0.0,
            "sample_rate_gsa_s": 200.0,
            "time_window_ns": 5.0,
            "settle_window_ns": 1.0,
            "settle_samples": 200,
            "photodetector_responsivity_a_w": 1.0,
            "lo_phase_reference_rad": 0.0,
            "lo_phase_quadrature_rad": float(np.pi / 2),
            "balanced_iq_formula": "I=-(P_Q+ - P_Q-)/2, Q=(P_I+ - P_I-)/2 with unit LO",
            "phase_alignment": "fixed 0/pi2 LO reference; no per-sample phase fitting",
            "fork_note": "INTERCONNECT Fork 1xN is an ideal coherent field copy, not a physical 1/N splitter",
        },
        "frequency_matrix_passed": bool(freq_case["passed"]),
        "records": records,
        "record_checks": [r["checks"] for r in records],
        "passed": time_domain_passed(freq_case["passed"], [r["checks"] for r in records]),
    }
    save_json(args.out / "time_domain.json", payload)
    print(f"time-domain overall: {'PASS' if payload['passed'] else 'FAIL'}", flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
