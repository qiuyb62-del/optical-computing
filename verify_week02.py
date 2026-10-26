"""用独立 QR 目标矩阵批量验收，生成可复现的第二周实验报告。"""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
from src.clements import decompose_clements
from src.mesh import propagate_mesh, reconstruct_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100, help="每种尺寸随机样本数")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "progress/02_clements_mesh/results.json")
    args = parser.parse_args()
    if args.samples < 1 or args.seed < 0:
        parser.error("samples 必须为正整数；seed 必须为非负整数")
    rng = np.random.default_rng(args.seed)
    rows = []
    for n in [2, 4, 8]:
        k = np.arange(n)
        matrices = [np.eye(n, dtype=complex), np.eye(n, dtype=complex)[::-1],
                    np.diag(np.exp(1j * np.linspace(-np.pi, 2 * np.pi, n))),
                    np.exp(2j * np.pi * np.outer(k, k) / n) / np.sqrt(n)]
        for _ in range(args.samples):
            z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            q, r = np.linalg.qr(z)
            d = np.diag(r)
            matrices.append(q * (d / np.abs(d))[None, :])
        errors = []
        for U in matrices:
            ops, phases = decompose_clements(U)
            Uhat = reconstruct_mesh(ops, phases)
            x = rng.normal(size=n) + 1j * rng.normal(size=n)
            y, reference = propagate_mesh(x, ops, phases), U @ x
            p_in, p_out = np.vdot(x, x).real, np.vdot(y, y).real
            errors.append([
                np.linalg.norm(Uhat - U, "fro") / np.linalg.norm(U, "fro"),
                np.linalg.norm(y - reference) / np.linalg.norm(reference),
                abs(p_out - p_in) / p_in,
                np.linalg.norm(Uhat.conj().T @ Uhat - np.eye(n), "fro"),
            ])
        maxima = np.max(errors, axis=0)
        row = {"size": n, "mzi_count": n * (n - 1) // 2,
               "random_samples": args.samples, "special_samples": 4,
               "max_matrix_relative_error": float(maxima[0]),
               "max_vector_relative_error": float(maxima[1]),
               "max_power_relative_error": float(maxima[2]),
               "max_unitarity_absolute_error": float(maxima[3]),
               "passed": bool(np.all(np.isfinite(maxima)) and np.all(maxima <= 1e-10))}
        rows.append(row)
        print(f"N={n}: {len(matrices)} 个矩阵，{row['mzi_count']} 个 MZI；"
              f"最大重建相对误差 {maxima[0]:.3e}；{'PASS' if row['passed'] else 'FAIL'}")
    passed = all(row["passed"] for row in rows)
    report = {"python": platform.python_version(), "numpy": np.__version__,
              "seed": args.seed, "tolerance": 1e-10,
              "special_cases": ["identity", "reversal", "diagonal_phase", "fourier"],
              "global_phase_alignment": False, "sizes": rows, "passed": passed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("报告:", args.output)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
