"""Publication-style standalone charts from recorded AI experiments."""
from pathlib import Path
import numpy as np


def save_plots(report, records, checkpoints, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "optical-ai-week05",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

    def save(fig, name):
        svg = output / f"{name}.svg"
        fig.savefig(svg, metadata={"Date": None})
        # Matplotlib emits trailing spaces inside path attributes; keep diffs clean.
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        fig.savefig(output / f"{name}.png", dpi=180, metadata={"Software": "Matplotlib"})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    runs = report["runs"]
    x = np.arange(len(runs))
    groups = [([100 * r["digital"]["accuracy"] for r in runs], "Digital", "#334155"),
              ([100 * r["ideal_optical"]["accuracy"] for r in runs], "Ideal optical", "#0284c7"),
              ([100 * r["conditions"][-1]["accuracy"]["mean"] for r in runs], "Combined errors (mean)", "#d97706")]
    for j, (values, label, color) in enumerate(groups):
        axes[0].bar(x + (j-1)*.24, values, .23, label=label, color=color)
    axes[0].set(xticks=x, xticklabels=[str(r["seed"])[-2:] for r in runs], ylim=(0, 105),
                ylabel="Test accuracy (%)", xlabel="Model / split seed (last two digits)",
                title="Same weights and test examples")
    axes[0].legend(loc="lower left", fontsize=9)
    noisy = [r for r in records if r["condition"] == "combined"]
    axes[1].scatter([100*r["linear_relative_error"] for r in noisy],
                    [r["accuracy_delta_pp"] for r in noisy], alpha=.5, s=20, color="#d97706")
    axes[1].axhline(0, color="#64748b", lw=1)
    axes[1].set(xlabel="Complex first-layer relative error (%)",
                ylabel="Accuracy change from digital (percentage points)",
                title="Numerical error and classification are distinct")
    fig.suptitle("Iris optical inference verification — no calibration", fontsize=15)
    save(fig, "accuracy")

    labels = {"phase_0.01": "Phase SD 0.01 rad", "phase_0.03": "Phase SD 0.03 rad",
              "phase_0.1": "Phase SD 0.10 rad", "bits_4": "4-bit phase control",
              "bits_6": "6-bit phase control", "bits_8": "8-bit phase control",
              "loss_0.1": "0.1 dB loss per MZI", "coupler_0.49": "Coupler through power 0.49",
              "combined": "Combined errors"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained", sharey=True)
    for ax, metric, factor, title in ((axes[0], "accuracy", 100, "Test accuracy (%)"),
                                      (axes[1], "accuracy_delta_pp", 1, "Change from digital (pp)")):
        for i, condition in enumerate(report["conditions"]):
            s = condition[metric]
            ax.hlines(i, factor*s["p05"], factor*s["p95"], color="#94a3b8", lw=3)
            ax.plot(factor*s["mean"], i, "o", color="#0284c7")
        ax.set(yticks=range(len(labels)), yticklabels=[labels[c["name"]] for c in report["conditions"]],
               xlabel=title)
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 102)
    axes[0].axvline(100*report["digital_accuracy"]["mean"], color="#334155", ls="--", label="Digital mean")
    axes[0].legend(loc="lower left")
    axes[1].axvline(0, color="#334155", ls="--")
    fig.suptitle("Fixed condition sweep: dots = means; ranges = empirical 5th–95th percentiles", fontsize=13)
    save(fig, "robustness")

    # Predeclared first model / first trial, never chosen by observed accuracy.
    first = runs[0]
    example = next(r for r in records if r["model_seed"] == first["seed"] and r["condition"] == "combined" and r["trial"] == 0)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.2), layout="constrained")
    class_names = ["setosa", "versicolor", "virginica"]
    for ax, values, title in zip(axes, [first["digital"], first["ideal_optical"], example],
                                 ["Digital", "Ideal optical", "Combined errors"]):
        cm = np.asarray(values["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=10)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j]>5 else "#0f172a", fontsize=14)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=class_names, yticklabels=class_names,
               xlabel="Predicted class", title=f"{title}\nAccuracy {values['accuracy']:.1%}")
        ax.tick_params(axis="x", labelrotation=25)
    axes[0].set_ylabel("True class")
    fig.suptitle(f"Same 30 test rows — first model seed {first['seed']}, device trial 0", fontsize=13)
    save(fig, "confusion")

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for cp in checkpoints:
        h = cp["training"]["history"]
        line, = ax.plot([p["epoch"] for p in h], [p["validation_loss"] for p in h],
                        label=str(cp["seed"]))
        ax.plot(cp["training"]["best_epoch"], cp["training"]["best_validation_loss"],
                "o", color=line.get_color())
    ax.set(xlabel="Epoch", ylabel="Validation cross-entropy", title="Checkpoint selection uses validation data only")
    ax.legend(title="Model / split seed")
    ax.grid(alpha=.2)
    save(fig, "training")
