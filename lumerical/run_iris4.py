"""Measure the saved Iris 4x4 first-layer weight in Lumerical and replay its test split."""
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
from src.ai_data import load_iris  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--out", type=Path, default=ROOT / "progress/07_lumerical/iris4")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / "progress/05_ai_extension/models" / f"seed_{args.seed}.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    W1 = np.asarray(model["weights"]["W1"], dtype=complex)
    b1 = np.asarray(model["weights"]["b1"], dtype=float)
    W2 = np.asarray(model["weights"]["W2"], dtype=float)
    b2 = np.asarray(model["weights"]["b2"], dtype=float)
    X, y = load_iris()
    split = model["split_indices"]["test"]
    mean = np.asarray(model["standardizer"]["mean"], dtype=float)
    scale = np.asarray(model["standardizer"]["scale"], dtype=float)
    X_test = (X[split] - mean) / scale
    y_test = y[split]

    case = measure_svd(W1, args.out, f"seed_{args.seed}")
    save_json(args.out / f"seed_{args.seed}_matrix.json", case)
    H = np.asarray(case["H"]["real"]) + 1j * np.asarray(case["H"]["imag"])
    s = float(case["recovery_scale"])
    digital_linear = X_test @ W1.T
    optical_linear = s * (X_test @ H.T)
    digital_hidden = np.maximum(digital_linear + b1, 0.0)
    optical_hidden = np.maximum(optical_linear.real + b1, 0.0)
    digital_logits = digital_hidden @ W2.T + b2
    optical_logits = optical_hidden @ W2.T + b2
    digital_pred = digital_logits.argmax(1)
    optical_pred = optical_logits.argmax(1)
    norm = float(np.linalg.norm(digital_linear))
    payload = {
        "stage": "lumerical_iris4_hybrid",
        "label": "Lumerical transmission-matrix-extracted hybrid inference",
        "seed": args.seed,
        "model_file": f"progress/05_ai_extension/models/seed_{args.seed}.json",
        "circuit_file": f"seed_{args.seed}.icp",
        "test_samples": int(len(y_test)),
        "lumerical_matrix_relative_error": case["relative_error"],
        "recovery_scale": s,
        "digital_accuracy": float(np.mean(digital_pred == y_test)),
        "lumerical_hybrid_accuracy": float(np.mean(optical_pred == y_test)),
        "prediction_agreement": float(np.mean(optical_pred == digital_pred)),
        "linear_relative_error": float(np.linalg.norm(optical_linear - digital_linear) / norm),
        "imaginary_leakage_ratio": float(np.linalg.norm(optical_linear.imag) / norm),
        "test_indices": [int(v) for v in split],
        "predictions": {"digital": digital_pred.tolist(), "lumerical": optical_pred.tolist(),
                        "labels": y_test.tolist()},
        "note": "Only the 4x4 first-layer weight matrix was extracted from Lumerical; bias, ReLU and the 4->3 head stay digital.",
    }
    save_json(args.out / "hybrid_inference.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0 if case["passed"] and case.get("restored_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
