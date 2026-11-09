"""MNIST stage 6. Keep Iris stage 5 intact; see docs/mnist-validation.md."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from src.ai_data import integer
from src.mnist_data import load_mnist, training_split, PixelPCA, RAW_DIR
from src.mnist_training import MNISTTrainingConfig, train_mnist
from src.ai_inference import digital_forward, optical_forward, classification_metrics, compare_forward
from src.batch_optics import optical_forward_batch
from src.svd_matmul import compile_svd, reconstruct_svd
from src.nonideal import NonidealConfig, realize_nonideal
from verify_week05 import dump_json, save_csv, stats, summarize

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "progress/06_mnist"
DEFAULT_SEEDS = (20261109, 20261110, 20261111)
SPLIT_SEED = 20261109


def conditions():
    return [
        ("phase_0.01", NonidealConfig(phase_std_rad=0.01)),
        ("phase_0.1", NonidealConfig(phase_std_rad=0.1)),
        ("bits_6", NonidealConfig(phase_bits=6)),
        ("combined", NonidealConfig(phase_std_rad=0.01, loss_db_per_mzi=0.1,
                                   phase_bits=8, coupler_through=0.49)),
    ]


def device_seed(model_seed, trial):
    return int(np.random.SeedSequence([model_seed, 0xA2, trial]).generate_state(1, dtype=np.uint64)[0])


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_metrics(target, logits):
    result = classification_metrics(target, logits)
    predictions = np.asarray(result.pop("predictions"), dtype=np.uint8)
    return result, predictions


def run_experiments(*, output_dir=DEFAULT_OUTPUT, raw_dir=RAW_DIR, download=False,
                    seeds=DEFAULT_SEEDS, trials=10, config=None, verbose=False):
    output = Path(output_dir)
    seeds = tuple(integer(s, "model seed") for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s >= 2**63 for s in seeds):
        raise ValueError("model seeds must be unique, nonempty integers less than 2**63")
    trials = integer(trials, "trials", 1)
    config = config or MNISTTrainingConfig()
    output.mkdir(parents=True, exist_ok=True)
    images, y, test_images, target, provenance = load_mnist(raw_dir, download=download)
    split = training_split(y, SPLIT_SEED)
    if verbose:
        print("Fitting PCA on 50,000 training images only...", flush=True)
    pca = PixelPCA.fit(images[split["train"]])
    pca.save(output / "preprocessor.npz")
    np.savez_compressed(output / "split_indices.npz", **split)
    train = pca.transform(images[split["train"]])
    validation = pca.transform(images[split["validation"]])
    test = pca.transform(test_images)
    # Selected by original order and label only, never correctness.
    demo_ids = np.concatenate([np.flatnonzero(target == digit)[:2] for digit in range(10)])
    np.savez_compressed(output / "demo_samples.npz", images=test_images[demo_ids],
                        labels=target[demo_ids], test_indices=demo_ids)
    artifacts = {name: file_hash(output / name)
                 for name in ("preprocessor.npz", "split_indices.npz", "demo_samples.npz")}
    dump_json(output / "data_provenance.json", {"resources": provenance, "artifacts_sha256": artifacts})
    runs, records, checkpoints = [], [], []
    predictions = {"test_labels": target.astype(np.uint8), "test_indices": np.arange(len(target))}
    for seed in seeds:
        if verbose:
            print(f"Training seed={seed}...", flush=True)
        weights, training = train_mnist(train, y[split["train"]], validation, y[split["validation"]],
                                         seed=seed, config=config)
        circuit = compile_svd(weights.W1)
        checkpoint = {
            "schema_version": 1, "dataset": "MNIST", "seed": seed,
            "preprocessor_sha256": artifacts["preprocessor.npz"],
            "split_sha256": artifacts["split_indices.npz"],
            "weights": weights.to_dict(), "ideal_circuit": circuit.to_dict(), "training": training}
        checkpoints.append(checkpoint)
        checkpoint_path = f"models/seed_{seed}.json"
        dump_json(output / checkpoint_path, checkpoint)
        reference = digital_forward(test, weights)
        digital, pred = compact_metrics(target, reference["logits"])
        predictions[f"digital_{seed}"] = pred
        ideal_out = optical_forward_batch(test, weights, circuit)
        ideal, predictions[f"ideal_{seed}"] = compact_metrics(target, ideal_out["logits"])
        comparison = compare_forward(ideal_out, reference)
        zero = optical_forward_batch(test, weights, realize_nonideal(circuit, seed=device_seed(seed, 0)))
        zero_comparison = compare_forward(zero, reference)
        # Independent scalar propagation checks catch batching/order/transpose errors.
        serial = optical_forward(test[:20], weights, circuit)
        W = np.asarray(weights.W1)
        matrix_error = float(np.linalg.norm(reconstruct_svd(circuit) - W) / np.linalg.norm(W))
        checks = {
            "torch_export_matches_numpy": training["torch_numpy_max_absolute_error"] < 1e-10,
            "ideal_matrix_matches": matrix_error < 1e-10,
            "ideal_linear_matches": bool(np.allclose(ideal_out["linear"], reference["linear"], atol=1e-10, rtol=1e-10)),
            "ideal_logits_match": bool(np.allclose(ideal_out["logits"], reference["logits"], atol=1e-10, rtol=1e-10)),
            "ideal_predictions_match": comparison["prediction_agreement"] == 1.0,
            "zero_error_linear_matches": bool(np.allclose(zero["linear"], reference["linear"], atol=1e-10, rtol=1e-10)),
            "zero_error_predictions_match": zero_comparison["prediction_agreement"] == 1.0,
            "batch_matches_scalar": bool(np.allclose(ideal_out["linear"][:20], serial["linear"], atol=1e-10, rtol=1e-10)),
            "ideal_passive": ideal_out["passivity_passed"], "zero_error_passive": zero["passivity_passed"],
        }
        run = {"seed": seed, "checkpoint": checkpoint_path,
               "training": {k: v for k, v in training.items() if k != "history"},
               "digital": digital, "ideal_optical": ideal, "ideal_comparison": comparison,
               "zero_error_comparison": zero_comparison, "ideal_matrix_relative_error": matrix_error,
               "checks": checks, "conditions": []}
        for name, cfg in conditions():
            condition_records = []
            for trial in range(trials):
                dev_seed = device_seed(seed, trial)
                device = realize_nonideal(circuit, cfg, seed=dev_seed)
                optical = optical_forward_batch(test, weights, device)
                classification, pred = compact_metrics(target, optical["logits"])
                key = f"noisy_{seed}_{name}_{trial}"
                predictions[key] = pred
                record = {"model_seed": seed, "condition": name, "trial": trial, "device_seed": dev_seed,
                          "prediction_key": key, **classification, **compare_forward(optical, reference),
                          "accuracy_delta_pp": 100 * (classification["accuracy"] - digital["accuracy"]),
                          "passivity_passed": optical["passivity_passed"]}
                condition_records.append(record)
                records.append(record)
            summary = {"name": name, "config": asdict(cfg), **summarize(condition_records)}
            run["conditions"].append(summary)
            if verbose:
                print(f"  {name}: mean accuracy={summary['accuracy']['mean']:.2%}", flush=True)
        run["passed"] = all(checks.values()) and all(r["passivity_passed"] for r in run["conditions"])
        runs.append(run)
        if verbose:
            print(f"seed={seed}: digital={digital['accuracy']:.2%}, ideal={ideal['accuracy']:.2%}, "
                  f"best epoch={training['best_epoch']}; checks={run['passed']}", flush=True)
    np.savez_compressed(output / "predictions.npz", **predictions)
    report = {
        "schema_version": 1, "experiment": "MNIST stage 6, frozen optical first layer; Iris retained",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": runs[0]["training"]["torch_version"], "system": platform.system(),
                        "machine": platform.machine(), "device": "cpu", "dtype": "float64"},
        "dataset": {"name": "MNIST", "training": 60000, "official_test": 10000,
                    "image_shape": [28, 28], "classes": list(range(10)), "resources": provenance},
        "protocol": {
            "split": [50000, 10000, 10000], "split_seed": SPLIT_SEED,
            "validation_selection": "first 1000 per class after seeded permutation of official training set",
            "preprocessing": "pixels/255; training-only centered PCA16; no whitening or extra standardization",
            "architecture": [16, 16, 10], "optical_operation": "first 16x16 weight only",
            "mzis_per_mesh": 120, "mzis_total": 240, "amplitude_attenuators": 16,
            "readout": "fixed-reference real quadrature after digital scale recovery",
            "digital_operations": ["PCA", "scale recovery", "bias", "ReLU", "16-to-10 head", "argmax"],
            "training": asdict(config), "model_seeds": seeds, "trials_per_condition": trials,
            "checkpoint_selection": "validation loss only", "test_usage": "evaluation only; no tuning",
            "device_seed_rule": "SeedSequence([model_seed, 0xA2, trial]) uint64",
            "device_reused_for_all_images": True, "paired_conditions": True,
            "calibration": False, "noise_retraining": False,
            "cross_platform_bitwise_reproducibility": False,
            "limitations": "Shared test set; pure quantization repeats are duplicates; empirical spreads are not confidence intervals",
        },
        "pca_explained_variance_ratio": pca.explained_variance_ratio.tolist(),
        "pca_retained_variance": float(pca.explained_variance_ratio.sum()),
        "artifacts_sha256": {**artifacts, "predictions.npz": file_hash(output / "predictions.npz")},
        "counts": {"trained_models": len(seeds), "conditions": len(conditions()),
                   "nonideal_configurations": len(records), "unique_test_images": len(target),
                   "nonideal_test_predictions": len(records) * len(target)},
        "digital_accuracy": stats([r["digital"]["accuracy"] for r in runs]),
        "ideal_accuracy": stats([r["ideal_optical"]["accuracy"] for r in runs]),
        "conditions": [{"name": name, "config": asdict(cfg),
                        **summarize([r for r in records if r["condition"] == name])}
                       for name, cfg in conditions()],
        "runs": runs, "passed": all(r["passed"] for r in runs),
        "pass_meaning": "numerical equivalence and passivity, no noisy accuracy threshold",
    }
    dump_json(output / "results.json", report)
    save_csv(records, output / "devices.csv")
    return report, records, checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the four checksum-verified official IDX files")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    custom = args.seeds != list(DEFAULT_SEEDS) or args.trials != 10 or args.max_epochs != 50 or args.patience != 8
    if custom and args.output_dir.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error("For development settings, supply a separate --output-dir, e.g. scratch_mnist")
    report, records, checkpoints = run_experiments(
        output_dir=args.output_dir, raw_dir=args.raw_dir, download=args.download,
        seeds=args.seeds, trials=args.trials,
        config=MNISTTrainingConfig(max_epochs=args.max_epochs, patience=args.patience), verbose=True)
    from demo_week06 import replay
    from src.mnist_plots import save_plots
    example = replay(args.output_dir, seed=args.seeds[0])
    dump_json(args.output_dir / "example.json", example)
    save_plots(report, records, checkpoints, example, args.output_dir)
    print(f"MNIST verification {'PASS' if report['passed'] else 'FAIL'}: {report['counts']}")
    print(f"Saved to {args.output_dir}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
