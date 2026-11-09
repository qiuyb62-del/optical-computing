"""Offline MNIST demo. No training, raw dataset, PyTorch or network required."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from src.mnist_data import PixelPCA
from src.ai_inference import ClassifierWeights, digital_forward, classification_metrics, compare_forward
from src.batch_optics import optical_forward_batch
from src.svd_matmul import compile_svd
from src.nonideal import realize_nonideal
from src.svd_validation import complex_data
from verify_week06 import DEFAULT_OUTPUT, conditions, device_seed, file_hash
from verify_week05 import dump_json


def replay(folder=DEFAULT_OUTPUT, *, seed=20261109):
    folder = Path(folder)
    checkpoint = json.loads((folder / f"models/seed_{seed}.json").read_text(encoding="utf-8"))
    if checkpoint["schema_version"] != 1 or checkpoint["dataset"] != "MNIST":
        raise ValueError("not a supported MNIST checkpoint")
    if file_hash(folder / "preprocessor.npz") != checkpoint["preprocessor_sha256"]:
        raise ValueError("PCA checkpoint checksum mismatch")
    provenance = json.loads((folder / "data_provenance.json").read_text(encoding="utf-8"))
    if file_hash(folder / "demo_samples.npz") != provenance["artifacts_sha256"]["demo_samples.npz"]:
        raise ValueError("demo sample checksum mismatch")
    pca = PixelPCA.load(folder / "preprocessor.npz")
    with np.load(folder / "demo_samples.npz", allow_pickle=False) as data:
        X = pca.transform(data["images"])
        target, ids = data["labels"], data["test_indices"]
    weights = ClassifierWeights(**checkpoint["weights"])
    circuit = compile_svd(weights.W1)
    cfg = dict(conditions())["combined"]
    dev_seed = device_seed(seed, 0)
    device = realize_nonideal(circuit, cfg, seed=dev_seed)
    reference = digital_forward(X, weights)
    outputs = {"digital": reference, "ideal_optical": optical_forward_batch(X, weights, circuit),
               "nonideal_combined": optical_forward_batch(X, weights, device)}
    result = {"selection": "first two official test images of each digit; not selected by predictions",
              "scope": "20 illustrative images, not the full-test accuracy estimate",
              "model_seed": seed, "device_trial": 0, "device_seed": dev_seed, "config": asdict(cfg),
              "test_indices": ids.tolist(), "test_labels": target.tolist(), "pca_features": X.tolist(),
              "versions": {}}
    for name, out in outputs.items():
        result["versions"][name] = {
            **classification_metrics(target, out["logits"]), **compare_forward(out, reference),
            "linear": complex_data(out["linear"]), "logits": out["logits"].tolist()}
        if name != "digital":
            result["versions"][name].update(raw_optical=complex_data(out["raw_optical"]),
                                             powers=out["powers"].tolist(), passivity_passed=out["passivity_passed"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--output", type=Path, help="Optionally save the replay, without overwriting by default")
    args = parser.parse_args()
    result = replay(args.folder, seed=args.seed)
    print("MNIST: 28x28 image -> digital PCA16 -> optical 16x16 -> digital bias/ReLU/head")
    print("Iris remains in demo_week05.py. This is the next research stage.")
    print("Twenty fixed illustration images, NOT the complete test accuracy estimate.")
    for j, idx in enumerate(result["test_indices"]):
        pred = [v["predictions"][j] for v in result["versions"].values()]
        print(f"test #{idx:4d}, truth={result['test_labels'][j]}: digital/ideal/noisy = {pred}")
    report_path = args.folder / "results.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        combined = next(c for c in report["conditions"] if c["name"] == "combined")
        print("Saved full official-test results (10,000 images; means over recorded models/devices):")
        print(f"digital={report['digital_accuracy']['mean']:.2%}, "
              f"ideal={report['ideal_accuracy']['mean']:.2%}, combined={combined['accuracy']['mean']:.2%}")
    if args.output:
        dump_json(args.output, result)
    print(f"View handwritten examples: {args.folder / 'examples.png'}")


if __name__ == "__main__":
    main()
