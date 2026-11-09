"""Train five frozen Iris classifiers and evaluate paired optical devices.

Run from any working directory. Defaults implement the predeclared protocol in
docs/ai-validation.md. Quick checks must use a separate --output-dir.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import numpy as np
from src.ai_data import DATA_SHA256, CLASS_NAMES, load_iris, stratified_split, Standardizer, integer
from src.ai_inference import digital_forward, optical_forward, classification_metrics, compare_forward
from src.ai_training import TrainingConfig, train_classifier
from src.svd_matmul import compile_svd, reconstruct_svd
from src.nonideal import NonidealConfig, realize_nonideal

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "progress/05_ai_extension"
DEFAULT_SEEDS = (20261109, 20261110, 20261111, 20261112, 20261113)


def conditions():
    return [
        ("phase_0.01", NonidealConfig(phase_std_rad=0.01)),
        ("phase_0.03", NonidealConfig(phase_std_rad=0.03)),
        ("phase_0.1", NonidealConfig(phase_std_rad=0.1)),
        ("bits_4", NonidealConfig(phase_bits=4)),
        ("bits_6", NonidealConfig(phase_bits=6)),
        ("bits_8", NonidealConfig(phase_bits=8)),
        ("loss_0.1", NonidealConfig(loss_db_per_mzi=0.1)),
        ("coupler_0.49", NonidealConfig(coupler_through=0.49)),
        ("combined", NonidealConfig(phase_std_rad=0.01, loss_db_per_mzi=0.1,
                                   phase_bits=8, coupler_through=0.49)),
    ]


def device_seed(model_seed, trial):
    return int(np.random.SeedSequence([model_seed, 0xA1, trial]).generate_state(1, dtype=np.uint64)[0])


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("summary values must be finite and nonempty")
    return {"count": int(values.size), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)), "p95": float(np.quantile(values, 0.95)),
            "min": float(values.min()), "max": float(values.max())}


def summarize(records):
    keys = ("accuracy", "accuracy_delta_pp", "linear_relative_error", "imaginary_leakage_ratio",
            "hidden_relative_error", "logit_relative_error", "cross_entropy", "prediction_agreement")
    result = {key: stats([r[key] for r in records]) for key in keys}
    result["summed_confusion_matrix"] = np.sum([r["confusion_matrix"] for r in records], axis=0).tolist()
    result["passivity_passed"] = all(r["passivity_passed"] for r in records)
    return result


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_experiments(*, seeds=DEFAULT_SEEDS, trials=30, config=None, output_dir=None, verbose=False):
    seeds = tuple(integer(s, "seed") for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    trials = integer(trials, "trials", 1)
    config = config or TrainingConfig()
    output_dir = Path(output_dir) if output_dir is not None else None
    X, y = load_iris()
    runs, records, checkpoints = [], [], []
    all_passed = True
    for seed in seeds:
        split = stratified_split(y, seed)
        scaler = Standardizer.fit(X[split["train"]])
        train, validation, test = (scaler.transform(X[split[name]]) for name in ("train", "validation", "test"))
        weights, training = train_classifier(train, y[split["train"]], validation, y[split["validation"]],
                                             seed=seed, config=config)
        # Compilation is once per trained model, never once per input or device.
        circuit = compile_svd(weights.W1)
        checkpoint = {"schema_version": 1, "dataset_sha256": DATA_SHA256, "seed": seed,
                      "split_indices": {k: v.tolist() for k, v in split.items()},
                      "class_names": CLASS_NAMES, "standardizer": scaler.to_dict(),
                      "weights": weights.to_dict(), "ideal_circuit": circuit.to_dict(), "training": training}
        checkpoints.append(checkpoint)
        checkpoint_path = f"models/seed_{seed}.json"
        if output_dir:
            dump_json(output_dir / checkpoint_path, checkpoint)
        reference = digital_forward(test, weights)
        target = y[split["test"]]
        digital = classification_metrics(target, reference["logits"])
        ideal_out = optical_forward(test, weights, circuit)
        ideal = classification_metrics(target, ideal_out["logits"])
        ideal_comparison = compare_forward(ideal_out, reference)
        zero = optical_forward(test, weights, realize_nonideal(circuit, seed=device_seed(seed, 0)))
        zero_comparison = compare_forward(zero, reference)
        W = np.asarray(weights.W1)
        matrix_error = float(np.linalg.norm(reconstruct_svd(circuit) - W) / np.linalg.norm(W))
        checks = {
            "torch_export_matches_numpy": training["torch_numpy_max_absolute_error"] < 1e-10,
            "ideal_matrix_matches": matrix_error < 1e-10,
            "ideal_linear_matches": bool(np.allclose(ideal_out["linear"], reference["linear"], atol=1e-10, rtol=1e-10)),
            "ideal_logits_match": bool(np.allclose(ideal_out["logits"], reference["logits"], atol=1e-10, rtol=1e-10)),
            "ideal_predictions_match": ideal_comparison["prediction_agreement"] == 1.0,
            "zero_error_physical_matches": bool(np.allclose(zero["linear"], reference["linear"], atol=1e-10, rtol=1e-10)),
            "zero_error_predictions_match": zero_comparison["prediction_agreement"] == 1.0,
            "ideal_passive": ideal_out["passivity_passed"], "zero_error_passive": zero["passivity_passed"],
        }
        run = {"seed": seed, "checkpoint": checkpoint_path, "test_indices": split["test"].tolist(),
               "test_labels": target.tolist(), "training": {k: v for k, v in training.items() if k != "history"},
               "digital": digital, "ideal_optical": ideal, "ideal_comparison": ideal_comparison,
               "zero_error_comparison": zero_comparison, "ideal_matrix_relative_error": matrix_error,
               "checks": checks, "conditions": []}
        for name, nonideal_config in conditions():
            condition_records = []
            for trial in range(trials):
                dev_seed = device_seed(seed, trial)
                device = realize_nonideal(circuit, nonideal_config, seed=dev_seed)
                optical = optical_forward(test, weights, device)
                classification = classification_metrics(target, optical["logits"])
                record = {"model_seed": seed, "condition": name, "trial": trial,
                          "device_seed": dev_seed, **classification, **compare_forward(optical, reference),
                          "accuracy_delta_pp": 100 * (classification["accuracy"] - digital["accuracy"]),
                          "passivity_passed": optical["passivity_passed"]}
                condition_records.append(record)
                records.append(record)
            run["conditions"].append({"name": name, "config": asdict(nonideal_config), **summarize(condition_records)})
        run["passed"] = all(checks.values()) and all(c["passivity_passed"] for c in run["conditions"])
        all_passed &= run["passed"]
        runs.append(run)
        if verbose:
            combined = run["conditions"][-1]["accuracy"]["mean"]
            print(f"seed={seed}: digital={digital['accuracy']:.2%}, ideal={ideal['accuracy']:.2%}, "
                  f"combined mean={combined:.2%}; selected epoch={training['best_epoch']}; checks={run['passed']}", flush=True)
    aggregate = [{"name": name, "config": asdict(cfg),
                  **summarize([r for r in records if r["condition"] == name])}
                 for name, cfg in conditions()]
    report = {
        "schema_version": 1, "experiment": "Iris frozen first optical layer without calibration",
        "dataset": {"source": "https://archive.ics.uci.edu/dataset/53/iris", "file": "bezdekIris.data",
                    "sha256": DATA_SHA256, "samples": 150, "features": 4, "classes": CLASS_NAMES},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": runs[0]["training"]["torch_version"], "system": platform.system(),
                        "machine": platform.machine(), "device": "cpu", "dtype": "float64"},
        "protocol": {"architecture": [4, 4, 3], "optical_operation": "first 4x4 weight only",
                     "readout": "fixed-reference real quadrature of restored complex field",
                     "digital_operations": ["standardization", "bias", "ReLU", "4-to-3 head", "argmax"],
                     "split": [90, 30, 30], "split_method": "stratified per class 30/10/10",
                     "standardization_fit": "train only", "checkpoint_selection": "validation loss only",
                     "test_usage": "evaluation only; no hyperparameter selection",
                     "training": asdict(config), "model_seeds": seeds, "trials_per_condition": trials,
                     "device_seed_rule": "SeedSequence([model_seed, 0xA1, trial]) uint64",
                     "paired_devices": True, "calibration": False, "retraining_under_noise": False,
                     "cross_platform_bitwise_reproducibility": False,
                     "percentile_interpretation": "empirical distributions, not confidence intervals",
                     "sampling_limitation": "repeated splits overlap; repeated deterministic devices are duplicates; test set has 30 rows"},
        "counts": {"trained_models": len(seeds), "conditions": len(conditions()),
                   "nonideal_configurations": len(records), "nonideal_test_predictions": len(records) * 30,
                   "digital_test_predictions": len(seeds) * 30, "ideal_test_predictions": len(seeds) * 30},
        "digital_accuracy": stats([r["digital"]["accuracy"] for r in runs]),
        "ideal_accuracy": stats([r["ideal_optical"]["accuracy"] for r in runs]),
        "conditions": aggregate, "runs": runs, "passed": bool(all_passed),
        "pass_meaning": "numerical equivalence and passivity; no prescribed noisy accuracy threshold",
    }
    return report, records, checkpoints


def save_csv(records, path):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, list) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(max_epochs=args.max_epochs, patience=args.patience)
    report, records, checkpoints = run_experiments(seeds=args.seeds, trials=args.trials, config=config,
                                                   output_dir=args.output_dir, verbose=True)
    dump_json(args.output_dir / "results.json", report)
    save_csv(records, args.output_dir / "devices.csv")
    from src.ai_plots import save_plots
    save_plots(report, records, checkpoints, args.output_dir)
    print(f"AI verification {'PASS' if report['passed'] else 'FAIL'}: {report['counts']}")
    print(f"Saved to {args.output_dir}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
