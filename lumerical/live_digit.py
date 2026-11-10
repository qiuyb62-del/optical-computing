"""Single-image live digit recognition through the 16-channel optical layer.

The live path builds the 16x16 SVD circuit from the frozen W1, encodes the
PCA feature vector on the optical inputs, runs INTERCONNECT time-domain
coherent I/Q readout, and reconstructs the complex linear output.  It does not
compute H@x and label it as a live simulation.

The digital path is always computed as a baseline.  A cached-H mode is kept as
an explicitly labelled demo only; it is not the live result.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mnist_data import PixelPCA  # noqa: E402

MODEL_DIR = ROOT / "progress/06_mnist/models"
PCA_PATH = ROOT / "progress/06_mnist/preprocessor.npz"
DEFAULT_OUT = ROOT / "progress/07_lumerical/live_digit"
CACHED_H = ROOT / "progress/07_lumerical/mnist16/seed_20261109_matrix.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strict_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    tmp.replace(path)


def load_assets(seed: int = 20261109):
    model_path = MODEL_DIR / f"seed_{seed}.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"missing frozen model: {model_path}")
    if not PCA_PATH.is_file():
        raise FileNotFoundError(f"missing PCA: {PCA_PATH}")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("schema_version") != 1 or model.get("dataset") != "MNIST":
        raise ValueError("unsupported MNIST checkpoint")
    actual_pca_hash = _sha256(PCA_PATH)
    if actual_pca_hash != model.get("preprocessor_sha256"):
        raise ValueError("PCA checksum does not match the frozen model")
    pca = PixelPCA.load(PCA_PATH)
    weights = model["weights"]
    W1 = np.asarray(weights["W1"], dtype=float)
    if W1.shape != (16, 16) or not np.all(np.isfinite(W1)):
        raise ValueError("frozen W1 must be a finite 16x16 matrix")
    return model, pca


def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.linspace(0, image.shape[0] - 1, height).round().astype(int)
    cols = np.linspace(0, image.shape[1] - 1, width).round().astype(int)
    return image[rows[:, None], cols[None, :]]


def preprocess_canvas(pixels, width: int, height: int) -> tuple[np.ndarray, dict]:
    """Turn a canvas grayscale array into a centered 28x28 uint8 digit."""
    array = np.asarray(pixels, dtype=float)
    if array.size != int(width) * int(height):
        raise ValueError(f"pixel count {array.size} does not match {width}x{height}")
    gray = np.clip(array.reshape(int(height), int(width)), 0.0, 255.0)
    ink = 1.0 - gray / 255.0
    ink[ink < 0.04] = 0.0
    if float(ink.max()) < 0.15 or float(ink.sum()) < 20.0:
        raise ValueError("empty canvas; draw a digit first")
    ys, xs = np.nonzero(ink > 0.05)
    if ys.size == 0:
        raise ValueError("empty canvas; draw a digit first")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = ink[y0:y1, x0:x1]
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        raise ValueError("stroke is too small to recognize")
    scale = 20.0 / max(crop.shape)
    new_h = max(1, int(round(crop.shape[0] * scale)))
    new_w = max(1, int(round(crop.shape[1] * scale)))
    resized = _resize_nearest(crop, new_h, new_w)
    out = np.zeros((28, 28), dtype=np.float64)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    image = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return image, {"source_width": int(width), "source_height": int(height),
                   "bbox": [x0, y0, x1, y1], "ink_sum": float(ink.sum())}


def _softmax(logits):
    z = np.asarray(logits, dtype=float) - float(np.max(logits))
    e = np.exp(z)
    return e / e.sum()


def _forward_from_linear(linear, weights):
    linear = np.asarray(linear, dtype=float)
    b1 = np.asarray(weights["b1"], dtype=float)
    W2 = np.asarray(weights["W2"], dtype=float)
    b2 = np.asarray(weights["b2"], dtype=float)
    hidden = np.maximum(linear + b1, 0.0)
    logits = hidden @ W2.T + b2
    return hidden, logits, _softmax(logits)


def digital_forward(x, weights):
    W1 = np.asarray(weights["W1"], dtype=float)
    linear = W1 @ np.asarray(x, dtype=float)
    hidden, logits, prob = _forward_from_linear(linear, weights)
    return {"linear": linear, "hidden": hidden, "logits": logits,
            "probabilities": prob, "prediction": int(np.argmax(logits))}


def cached_forward(x, *, path: Path = CACHED_H):
    """Explicit cached-H demo: s * H @ x, not a live simulation."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"missing cached 16x16 H: {path}")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    H = np.asarray(data["H"]["real"]) + 1j * np.asarray(data["H"]["imag"])
    s = float(data["recovery_scale"])
    if H.shape != (16, 16) or not np.all(np.isfinite(H)):
        raise ValueError("cached H must be a finite 16x16 complex matrix")
    restored = s * (H @ np.asarray(x, dtype=complex))
    return {
        "raw_optical": H @ np.asarray(x, dtype=complex),
        "encoding_scale_g": 1.0,
        "recovery_scale_s": s,
        "restored": restored,
        "circuit_size": 16,
        "source": str(path),
    }


def live_forward(x, weights, seed: int = 20261109):
    """Run the 16-channel time-domain circuit and restore s*g*y."""
    from src.svd_matmul import compile_svd
    from lumerical.interconnect import app_version, new_app
    from lumerical.time_domain import build_td_svd, run_vector

    W1 = np.asarray(weights["W1"], dtype=float)
    circuit = compile_svd(W1)
    if circuit.size != len(x):
        raise ValueError("circuit size does not match input vector")
    x = np.asarray(x, dtype=complex)
    g = max(1.0, float(np.max(np.abs(x))))
    scaled = x / g
    app = new_app()
    try:
        build_start = time.perf_counter()
        testbench = build_td_svd(app, circuit, "LIVE16")
        build_seconds = time.perf_counter() - build_start
        run_start = time.perf_counter()
        y_measured, g_actual = run_vector(app, testbench, x)
        run_seconds = time.perf_counter() - run_start
        if not np.all(np.isfinite(y_measured)):
            raise RuntimeError("live INTERCONNECT returned non-finite complex output")
        restored = np.asarray(circuit.recovery_scale * g_actual * y_measured, dtype=complex)
        return {
            "raw_optical": y_measured,
            "encoding_scale_g": float(g_actual),
            "recovery_scale_s": float(circuit.recovery_scale),
            "restored": restored,
            "circuit_size": int(circuit.size),
            "mzi_count": int(len(circuit.input_ops) + len(circuit.output_ops)),
            "encoded_amplitude": np.abs(scaled).tolist(),
            "encoded_phase_rad": np.angle(scaled).tolist(),
            "build_seconds": float(build_seconds),
            "run_seconds": float(run_seconds),
            "lumerical_version": app_version(),
            "run_trace": [
                "compile_svd(W1) -> SVDCircuit(16)",
                "build_td_svd -> 240 MZI + 16 coherent I/Q receivers",
                "set_encoded_vector(x/g) -> 16 amplitudes + 16 phases",
                "app.run() -> INTERCONNECT transient",
                "read_iq -> 16 complex outputs",
                "restore s*g*y",
            ],
        }
    finally:
        app.close()


def recognize_pixels(pixels, width: int, height: int, *, seed=20261109,
                     simulate: bool = True, mode: str | None = None,
                     outdir: Path | None = None):
    started = time.perf_counter()
    if mode is None:
        mode = "live" if simulate else "digital"
    if mode not in ("live", "cached", "digital"):
        raise ValueError("mode must be live, cached or digital")
    simulate = mode == "live"
    model, pca = load_assets(seed)
    image, meta = preprocess_canvas(pixels, width, height)
    x = pca.transform(image[None, :, :])[0]
    if not np.all(np.isfinite(x)):
        raise RuntimeError("PCA produced non-finite features")
    digital = digital_forward(x, model["weights"])
    result = {
        "stage": "live_digit_single",
        "seed": int(seed),
        "model_file": f"progress/06_mnist/models/seed_{seed}.json",
        "preprocessor_file": "progress/06_mnist/preprocessor.npz",
        "model_sha256": _sha256(MODEL_DIR / f"seed_{seed}.json"),
        "pca_sha256": _sha256(PCA_PATH),
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "preprocess": meta,
        "image_28": image.tolist(),
        "features_x": x.tolist(),
        "feature_scale_g": float(max(1.0, np.max(np.abs(x)))),
        "digital": {
            "linear": digital["linear"].tolist(),
            "logits": digital["logits"].tolist(),
            "probabilities": digital["probabilities"].tolist(),
            "prediction": digital["prediction"],
        },
        "live_interconnect_called": False,
        "mode": mode,
        "optical": None,
        "status": "digital_only",
        "elapsed_seconds": None,
    }
    if mode != "digital":
        live = (live_forward(x, model["weights"], seed=seed)
                if mode == "live" else cached_forward(x))
        restored = live["restored"]
        optical = _forward_from_linear(restored.real, model["weights"])
        norm = max(float(np.linalg.norm(digital["linear"])), 1e-30)
        result["live_interconnect_called"] = mode == "live"
        result["optical"] = {
            "raw_optical": {"real": live["raw_optical"].real.tolist(),
                            "imag": live["raw_optical"].imag.tolist()},
            "encoding_scale_g": live["encoding_scale_g"],
            "recovery_scale_s": live["recovery_scale_s"],
            "restored_linear": {"real": restored.real.tolist(),
                                "imag": restored.imag.tolist()},
            "hidden": optical[0].tolist(),
            "logits": optical[1].tolist(),
            "probabilities": optical[2].tolist(),
            "prediction": int(np.argmax(optical[1])),
            "linear_relative_error": float(
                np.linalg.norm(restored.real - digital["linear"]) / norm),
            "imaginary_leakage_ratio": float(np.linalg.norm(restored.imag) / norm),
            "circuit_size": live["circuit_size"],
            "mzi_count": live.get("mzi_count"),
            "build_seconds": live.get("build_seconds"),
            "run_seconds": live.get("run_seconds"),
            "lumerical_version": live.get("lumerical_version"),
            "encoded_amplitude": live.get("encoded_amplitude"),
            "encoded_phase_rad": live.get("encoded_phase_rad"),
            "run_trace": live.get("run_trace"),
            "mode": mode,
        }
        result["status"] = "live_ok" if mode == "live" else "cached_H_demo"
    result["prediction_agreement"] = (
        None if result["optical"] is None
        else bool(result["optical"]["prediction"] == digital["prediction"]))
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = f"session_{result['image_sha256'][:10]}"
        _strict_json(outdir / f"{stem}.json", result)
        # Portable preview, no Pillow dependency.
        (outdir / f"{stem}.pgm").write_text(
            "P2\n28 28\n255\n" + " ".join(str(int(v)) for v in image.ravel()) + "\n",
            encoding="ascii")
        result["session_file"] = str(outdir / f"{stem}.json")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="optional 28x28 PGM/text image")
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--digital-only", action="store_true")
    parser.add_argument("--mode", choices=["live", "cached", "digital"], default="live")
    args = parser.parse_args()
    mode = "digital" if args.digital_only else args.mode
    if args.image:
        text = args.image.read_text(encoding="ascii").split()
        if text[:1] == ["P2"]:
            text = text[4:]
        pixels = np.asarray([int(v) for v in text], dtype=float)
        if pixels.size != 784:
            raise SystemExit("--image must contain 28x28 grayscale values")
        width = height = 28
    else:
        raise SystemExit("--image is required for CLI mode; use live_digit_ui.py for canvas")
    result = recognize_pixels(pixels, width, height, seed=args.seed,
                              mode=mode, outdir=args.out)
    print(json.dumps({
        "status": result["status"],
        "digital_prediction": result["digital"]["prediction"],
        "optical_prediction": None if result["optical"] is None
                              else result["optical"]["prediction"],
        "live_interconnect_called": result["live_interconnect_called"],
        "linear_relative_error": None if result["optical"] is None
                                 else result["optical"]["linear_relative_error"],
        "session_file": result.get("session_file"),
    }, indent=2))


if __name__ == "__main__":
    main()
