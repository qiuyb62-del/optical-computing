"""Build the 16-channel, 240-MZI first layer for MNIST and extract its S matrix.

The Lumerical circuit is measured once.  The resulting 16x16 complex H is then
applied in NumPy to the frozen PCA features of all 10,000 official test images;
that result is explicitly a hybrid inference, not 10,000 transient simulations.
A separate time-domain testbench directly validates selected Hx products.
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
from lumerical.interconnect import save_json  # noqa: E402
from src.mnist_data import PixelPCA, load_mnist  # noqa: E402


def load_model(seed):
    path = ROOT / "progress/06_mnist/models" / f"seed_{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen MNIST model: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def preflight(seed, *, download):
    """Validate data/model inputs before importing or running Lumerical."""
    model = load_model(seed)
    preprocessor_path = ROOT / "progress/06_mnist/preprocessor.npz"
    reference_path = ROOT / "progress/06_mnist/predictions.npz"
    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"missing MNIST PCA file: {preprocessor_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"missing MNIST reference predictions: {reference_path}")
    pca = PixelPCA.load(preprocessor_path)
    _, _, test_images, test_labels, _ = load_mnist(
        ROOT / "data/mnist/raw", download=download)
    if len(test_images) != 10000 or len(test_labels) != 10000:
        raise ValueError("official MNIST test set must contain 10,000 images and labels")
    weights = model["weights"]
    W1 = np.asarray(weights["W1"], dtype=float)
    if W1.shape != (16, 16) or not np.all(np.isfinite(W1)):
        raise ValueError("frozen W1 must be a finite 16x16 matrix")
    with np.load(reference_path) as reference:
        key = f"ideal_{seed}"
        if key not in reference:
            raise KeyError(f"reference predictions are missing {key}")
    return model, test_images, test_labels, pca


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--out", type=Path, default=ROOT / "progress/07_lumerical/mnist16")
    parser.add_argument("--download", action="store_true",
                        help="download official MNIST IDX files before starting Lumerical")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model, test_images, test_labels, pca = preflight(args.seed, download=args.download)
    W1 = np.asarray(model["weights"]["W1"], dtype=complex)
    print(f"Building 16x16 Lumerical SVD circuit for seed {args.seed}, "
          f"sigma_max={np.linalg.svd(W1, compute_uv=False)[0]:.6f}", flush=True)
    case = measure_svd(W1, args.out, f"seed_{args.seed}")
    save_json(args.out / f"seed_{args.seed}_matrix.json", case)
    print(f"circuit measured: passed={case['passed']} "
          f"relative={case['relative_error']:.3e}", flush=True)
    if not case["passed"] or not case.get("restored_passed", True):
        raise RuntimeError("Lumerical matrix validation failed; refusing hybrid inference")
    H = np.asarray(case["H"]["real"]) + 1j * np.asarray(case["H"]["imag"])
    s = float(case["recovery_scale"])
    test_idx = np.arange(len(test_images), dtype=np.int64)
    X = pca.transform(test_images)
    weights = model["weights"]
    W1_saved = np.asarray(weights["W1"], dtype=float)
    b1 = np.asarray(weights["b1"], dtype=float)
    W2 = np.asarray(weights["W2"], dtype=float)
    b2 = np.asarray(weights["b2"], dtype=float)
    y = np.asarray(test_labels, dtype=int)

    digital_linear = X @ W1_saved.T
    lumerical_linear = s * (X @ H.T)
    digital_hidden = np.maximum(digital_linear + b1, 0.0)
    lumerical_hidden = np.maximum(lumerical_linear.real + b1, 0.0)
    digital_logits = digital_hidden @ W2.T + b2
    lumerical_logits = lumerical_hidden @ W2.T + b2
    digital_pred = digital_logits.argmax(1)
    lumerical_pred = lumerical_logits.argmax(1)

    reference = np.load(ROOT / "progress/06_mnist/predictions.npz")
    saved_ideal = reference[f"ideal_{args.seed}"]
    norm = float(np.linalg.norm(digital_linear))
    payload = {
        "stage": "lumerical_mnist16_hybrid",
        "label": "Lumerical transmission-matrix-extracted hybrid inference",
        "seed": args.seed,
        "model_file": f"progress/06_mnist/models/seed_{args.seed}.json",
        "circuit_file": f"seed_{args.seed}.icp",
        "test_samples": int(len(y)),
        "lumerical_matrix_relative_error": case["relative_error"],
        "lumerical_matrix_absolute_error": case["absolute_error"],
        "recovery_scale": s,
        "digital_accuracy": float(np.mean(digital_pred == y)),
        "lumerical_hybrid_accuracy": float(np.mean(lumerical_pred == y)),
        "saved_python_ideal_accuracy": float(np.mean(saved_ideal == y)),
        "prediction_agreement_with_saved_ideal": float(np.mean(lumerical_pred == saved_ideal)),
        "prediction_agreement_with_digital": float(np.mean(lumerical_pred == digital_pred)),
        "linear_relative_error": float(np.linalg.norm(lumerical_linear - digital_linear) / norm),
        "imaginary_leakage_ratio": float(np.linalg.norm(lumerical_linear.imag) / norm),
        "logit_max_absolute_error": float(np.max(np.abs(lumerical_logits - digital_logits))),
        "note": "H was extracted from one Lumerical circuit; the 10,000 test images were not each simulated in Lumerical.",
    }
    (args.out / "hybrid_inference.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(args.out / "hybrid_predictions.npz",
                        labels=y, digital=digital_pred, lumerical=lumerical_pred,
                        test_indices=test_idx)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0 if case["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
