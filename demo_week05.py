"""Replay the first predeclared trained model. No PyTorch or network required."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from src.ai_data import DATA_SHA256, CLASS_NAMES, load_iris, Standardizer
from src.ai_inference import ClassifierWeights, digital_forward, optical_forward, classification_metrics, compare_forward
from src.svd_matmul import compile_svd
from src.nonideal import NonidealConfig, realize_nonideal
from src.svd_validation import complex_data
from verify_week05 import DEFAULT_OUTPUT, device_seed, dump_json


def replay(checkpoint):
    if checkpoint["schema_version"] != 1 or checkpoint["dataset_sha256"] != DATA_SHA256:
        raise ValueError("checkpoint schema or dataset mismatch")
    X, y = load_iris()
    split = checkpoint["split_indices"]
    ids = np.concatenate([np.asarray(split[k]) for k in ("train", "validation", "test")])
    if ids.dtype.kind not in "iu" or not np.array_equal(np.sort(ids), np.arange(150)):
        raise ValueError("checkpoint split must partition all 150 data rows exactly once")
    if [len(split[k]) for k in ("train", "validation", "test")] != [90, 30, 30]:
        raise ValueError("checkpoint split sizes must be 90/30/30")
    scaler = Standardizer(checkpoint["standardizer"]["mean"], checkpoint["standardizer"]["scale"])
    fitted = Standardizer.fit(X[split["train"]])
    if scaler != fitted:
        raise ValueError("checkpoint standardizer must be fitted to its training split")
    test = scaler.transform(X[split["test"]])
    target = y[split["test"]]
    weights = ClassifierWeights(**checkpoint["weights"])
    circuit = compile_svd(weights.W1)
    config = NonidealConfig(phase_std_rad=0.01, loss_db_per_mzi=0.1, phase_bits=8, coupler_through=0.49)
    seed = device_seed(checkpoint["seed"], 0)
    device = realize_nonideal(circuit, config, seed=seed)
    reference = digital_forward(test, weights)
    outputs = {"digital": reference, "ideal_optical": optical_forward(test, weights, circuit),
               "nonideal_combined": optical_forward(test, weights, device)}
    result = {"selection": "first predefined model seed and device trial zero; not selected by accuracy",
              "model_seed": checkpoint["seed"], "device_seed": seed, "config": asdict(config),
              "test_indices": split["test"], "test_labels": target.tolist(), "class_names": CLASS_NAMES,
              "raw_features": X[split["test"]].tolist(), "standardized_features": test.tolist(),
              "weights": weights.to_dict(), "ideal_circuit": circuit.to_dict(), "device": device.to_dict(),
              "versions": {}}
    for name, out in outputs.items():
        result["versions"][name] = {**classification_metrics(target, out["logits"]),
                                    **compare_forward(out, reference),
                                    "linear": complex_data(out["linear"]), "logits": out["logits"].tolist()}
        if name != "digital":
            result["versions"][name].update(raw_optical=complex_data(out["raw_optical"]),
                                             powers=out["powers"].tolist(), passivity_passed=out["passivity_passed"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT / "models/seed_20261109.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "example.json")
    args = parser.parse_args()
    result = replay(json.loads(args.model.read_text(encoding="utf-8")))
    dump_json(args.output, result)
    print("Iris: 4 features -> optical 4x4 weights -> digital bias/ReLU -> digital 3-class head")
    print("One fixed model, the same 30 test examples, no calibration or noise retraining.")
    for name, values in result["versions"].items():
        print(f"{name:20s}: accuracy={values['accuracy']:.2%}, "
              f"linear error={values['linear_relative_error']:.3e}, "
              f"agreement with digital={values['prediction_agreement']:.2%}")
    print("First five test examples (original zero-based row IDs):")
    for j, row in enumerate(result["test_indices"][:5]):
        pred = [CLASS_NAMES[result["versions"][name]["predictions"][j]].replace("Iris-", "") for name in result["versions"]]
        print(f"row {row:3d}: true={CLASS_NAMES[result['test_labels'][j]]}; digital / ideal / noisy = {' / '.join(pred)}")
    print(f"Full complex fields, predictions and device snapshot saved to {args.output}")


if __name__ == "__main__":
    main()
