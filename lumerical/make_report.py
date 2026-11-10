"""Generate the Lumerical-stage figures from the saved JSON results only."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "progress/07_lumerical"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mzi = load(OUT / "single_mzi/single_mzi.json")
    clements = load(OUT / "clements4/clements4.json")
    svd = load(OUT / "svd4/svd4.json")
    nonideal = load(OUT / "nonideal4/nonideal4.json")
    td = load(OUT / "svd4_td/time_domain.json")
    iris = load(OUT / "iris4/hybrid_inference.json")
    mnist = load(OUT / "mnist16/hybrid_inference.json")
    fde = load(OUT / "device_fde/device_fde.json")
    fdtd = load(OUT / "device_fdtd/device_fdtd.json")

    labels = ["single MZI", "Clements 4", "SVD 4", "SVD 4 nonideal",
              "time-domain I/Q", "Iris 4 hybrid", "MNIST 16 hybrid"]
    values = [
        max(c["relative_error"] for c in mzi["cases"]),
        max(c["relative_error"] for c in clements["cases"]),
        max(c["relative_error"] for c in svd["cases"] if c["relative_error"] is not None),
        max(c["relative_error"] for c in nonideal["cases"]),
        max(r["optical_relative_error"] for r in td["records"]),
        iris["lumerical_matrix_relative_error"],
        mnist["lumerical_matrix_relative_error"],
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = ["#3569a8"] * 5 + ["#2a9d8f", "#e76f51"]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xscale("log")
    ax.set_xlabel("maximum relative complex-field error (log scale)")
    ax.set_title("Lumerical 2025 R2 validation: measured complex fields")
    ax.axvline(1e-6, color="k", ls="--", lw=1, label="engineering target 1e-6")
    for bar, value in zip(bars, values[::-1]):
        ax.text(value * 1.25, bar.get_y() + bar.get_height() / 2,
                f"{value:.1e}", va="center", fontsize=8)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "summary_errors.png", dpi=180)
    fig.savefig(OUT / "summary_errors.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for record, color in zip(td["records"], colors):
        expected = np.asarray(record["expected_optical_Hx"]["real"]) + \
            1j * np.asarray(record["expected_optical_Hx"]["imag"])
        measured = np.asarray(record["measured_optical_output"]["real"]) + \
            1j * np.asarray(record["measured_optical_output"]["imag"])
        ax.scatter(expected.real, expected.imag, marker="o", facecolors="none",
                   edgecolors=color, s=55, label=f"vector {record['index']} expected")
        ax.scatter(measured.real, measured.imag, marker="x", color=color, s=35)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Re(E)")
    ax.set_ylabel("Im(E)")
    ax.set_title("Time-domain coherent I/Q readout (o expected, x measured)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "time_domain_iq.png", dpi=180)
    fig.savefig(OUT / "time_domain_iq.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = ["Iris digital", "Iris Lumerical", "MNIST digital", "MNIST Lumerical"]
    acc = [iris["digital_accuracy"] * 100, iris["lumerical_hybrid_accuracy"] * 100,
           mnist["digital_accuracy"] * 100, mnist["lumerical_hybrid_accuracy"] * 100]
    bars = ax.bar(names, acc, color=["#457b9d", "#2a9d8f", "#e76f51", "#f4a261"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("AI first layer: digital vs Lumerical-transmission hybrid")
    for bar, value in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%",
                ha="center", fontsize=9)
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ai_accuracy.png", dpi=180)
    fig.savefig(OUT / "ai_accuracy.svg")
    plt.close(fig)

    lam = np.asarray(fde["wavelength_um"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, label, color in (("neff_coarse_60", "60x60 cells", "#9ecae1"),
                              ("neff_fine_120", "120x120", "#4292c6"),
                              ("neff_finer_480", "480x480", "#08306b")):
        neff = [v["real"] for v in fde[key]]
        ax.plot(lam, neff, "o-", label=label, color=color)
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("FDE effective index")
    ax.set_title("MODE FDE convergence: Si 0.5 x 0.22 um on SiO2, TE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fde_neff.png", dpi=180)
    fig.savefig(OUT / "fde_neff.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, label, color in (("coarse", "mesh accuracy 2", "#e76f51"),
                              ("fine", "mesh accuracy 3", "#264653")):
        data = fdtd[key]
        ax.plot(data["wavelength_um"], data["T"], "o-", label=label, color=color)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="passive limit")
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("power transmission T")
    ax.set_title("2D FDTD diagnostic: facet ripple, not a converged device S-param")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fdtd_diagnostic.png", dpi=180)
    fig.savefig(OUT / "fdtd_diagnostic.svg")
    plt.close(fig)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
