"""Shareable MNIST figures. Only imported when plotting."""
from pathlib import Path
import numpy as np


def save_plots(report, records, checkpoints, example, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    with np.load(output / "demo_samples.npz", allow_pickle=False) as data:
        images = data["images"]
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for j, ax in enumerate(axes.flat):
        truth = example["test_labels"][j]
        pred = [v["predictions"][j] for v in example["versions"].values()]
        ax.imshow(images[j], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"True {truth} | D {pred[0]} / I {pred[1]} / N {pred[2]}",
                     color="#b03a2e" if pred[2] != truth else "#17365d", fontsize=10)
        ax.set_xlabel(f"Official test index {example['test_indices'][j]}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("MNIST through a 16-channel optical layer", fontsize=18, y=1.01)
    fig.text(.5, .01, f"D: digital   I: ideal optical   N: combined static errors | First two images per digit; seed {example['model_seed']}, trial 0",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, 1))
    save(fig, "examples")

    labels = ["Digital", "Ideal optical", "Phase 0.01 rad", "Phase 0.1 rad", "6-bit", "Combined"]
    values = [report["digital_accuracy"]["mean"], report["ideal_accuracy"]["mean"]]
    values += [c["accuracy"]["mean"] for c in report["conditions"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, 100*np.array(values), color=["#566573", "#2471a3", "#73c6b6", "#e5a04b", "#6c78ac", "#b65e73"])
    ax.set_ylim(0, 105); ax.set_ylabel("Accuracy on 10,000 test images (%)")
    ax.set_title("Frozen models: ideal equivalence and sensitivity to static device errors")
    for j, run in enumerate(report["runs"]):
        ax.scatter([0,1], [100*run["digital"]["accuracy"],100*run["ideal_optical"]["accuracy"]],
                   color="black", s=13, alpha=.5)
    for i, condition in enumerate(report["conditions"], 2):
        ys = [100*r["accuracy"] for r in records if r["condition"] == condition["name"]]
        ax.scatter(np.full(len(ys), i) + np.linspace(-.12,.12,len(ys)), ys, s=10, color="black", alpha=.35)
    maxima = [100*report["digital_accuracy"]["max"], 100*report["ideal_accuracy"]["max"]]
    maxima += [100*c["accuracy"]["max"] for c in report["conditions"]]
    for i, (value, top) in enumerate(zip(values, maxima)):
        ax.text(i, top + 2, f"{100*value:.2f}%", ha="center", va="bottom")
    fig.text(.5, .01, "Bars: means; dots: model/device runs. Shared test set; quantization-only trials are deterministic repeats.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    save(fig, "accuracy")

    run = report["runs"][0]
    combined = next(r for r in records if r["model_seed"] == run["seed"] and r["condition"] == "combined" and r["trial"] == 0)
    fig, axes = plt.subplots(1,3,figsize=(14,4.4), constrained_layout=True)
    matrices = [run["digital"]["confusion_matrix"],run["ideal_optical"]["confusion_matrix"],combined["confusion_matrix"]]
    vmax = max(np.max(m) for m in matrices)
    for ax, matrix, label in zip(axes, matrices, ["Digital", "Ideal optical", "Combined errors, trial 0"]):
        img = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit"); ax.set_title(label)
    fig.colorbar(img, ax=axes, shrink=.8, label="Number of test images")
    fig.suptitle(f"Same 10,000 test images | Fixed first model seed {run['seed']}")
    save(fig, "confusion")

    fig, axes = plt.subplots(1, len(checkpoints), figsize=(5*len(checkpoints),4), squeeze=False)
    for ax, ck in zip(axes.flat, checkpoints):
        h = ck["training"]["history"]
        for key, label in [("train_loss","Training"),("validation_loss","Validation")]:
            ax.plot([v["epoch"] for v in h], [v[key] for v in h], label=label)
        ax.axvline(ck["training"]["best_epoch"], linestyle="--", color="gray", label="Selected epoch")
        ax.set(xlabel="Epoch", ylabel="Cross-entropy", title=f"Seed {ck['seed']}")
        ax.legend()
    fig.tight_layout()
    save(fig, "training")
