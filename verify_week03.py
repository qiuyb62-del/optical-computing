"""第三阶段批量验收：独立生成一般矩阵，比较逐器件光路与 W @ x。"""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
from src.svd_matmul import compile_svd, trace_svd, reconstruct_svd
from src.svd_validation import TOLERANCE, NEAR_ZERO_RATIO, evaluate


def special_matrices(n):
    phases = np.exp(1j * np.linspace(-.8, 1.3, n))
    v = np.arange(1, n + 1) + 1j * np.arange(n, 0, -1)
    return [
        ("zero", np.zeros((n, n), dtype=complex)),
        ("identity", np.eye(n, dtype=complex)),
        ("signed_attenuation", np.diag(np.linspace(-.8, .6, n)).astype(complex)),
        ("complex_gain", np.diag(np.linspace(2, 4, n) * phases)),
        ("rank_one", np.outer(v, v.conj())),
        ("rank_deficient", np.diag(np.r_[0., np.arange(1, n)]).astype(complex)),
        ("ill_conditioned", np.diag(np.geomspace(1., 1e-12, n) * phases)),
    ]


def maximum_relative(records, key):
    values = [r[key]["relative_error"] for r in records if r[key]["criterion"] == "relative"]
    return max(values, default=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100, help="每种尺寸的随机矩阵数；实数/复数交替")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "progress/03_svd_matmul/results.json")
    args = parser.parse_args()
    if args.samples < 1 or args.seed < 0:
        parser.error("samples 必须为正整数；seed 必须为非负整数")
    rng = np.random.default_rng(args.seed)
    rows = []
    for n in [2, 4, 8]:
        matrices = special_matrices(n)
        for i in range(args.samples):
            W = rng.normal(size=(n, n)).astype(complex)
            if i % 2:
                W += 1j * rng.normal(size=(n, n))
            # 同时覆盖整体衰减与恢复比例大于 1 的情况；目标不由 MZI 构造。
            W *= [.01, 1., 100.][i % 3]
            matrices.append((f"random_{'complex' if i % 2 else 'real'}_{i:03d}", W))
        records = []
        for name, W in matrices:
            circuit = compile_svd(W)
            reconstructed = reconstruct_svd(circuit)
            # 对同一配置输入每个基向量、一个随机复向量和零向量。
            inputs = [(f"basis_{j}", x) for j, x in enumerate(np.eye(n, dtype=complex))]
            inputs += [("random", rng.normal(size=n) + 1j * rng.normal(size=n)),
                       ("zero", np.zeros(n, dtype=complex))]
            for input_name, x in inputs:
                metrics = evaluate(W, x, circuit, trace_svd(x, circuit), reconstructed)
                records.append({"matrix_case": name, "input_case": input_name, **metrics})
        near_zero = [r["vector"] for r in records if r["vector"]["criterion"] == "absolute_near_zero"]
        row = {"size": n, "mzi_count_total": n * (n - 1),
               "random_matrices": args.samples, "special_matrices": 7,
               "matrix_count": len(matrices), "vector_count": len(records),
               "max_matrix_relative_error": maximum_relative(records, "matrix"),
               "max_vector_relative_error": maximum_relative(records, "vector"),
               "max_optical_vector_relative_error": maximum_relative(records, "optical_vector"),
               "near_zero_vector_count": len(near_zero),
               "max_near_zero_vector_absolute_error": max(r["absolute_error"] for r in near_zero),
               "power_checks_passed": all(r["power_checks_passed"] for r in records),
               "passed": all(r["passed"] for r in records)}
        rows.append(row)
        print(f"N={n}: {len(matrices)} 个矩阵，{len(records)} 组输入；"
              f"最大矩阵/输出相对误差 {row['max_matrix_relative_error']:.3e} / "
              f"{row['max_vector_relative_error']:.3e}；{'PASS' if row['passed'] else 'FAIL'}")
        # 失败时保存定位信息，成功时保留汇总，避免重复存放几千条基向量结果。
        failures = [r for r in records if not r["passed"]]
        if failures:
            row["failures"] = failures
    report = {"python": platform.python_version(), "numpy": np.__version__,
              "seed": args.seed, "tolerance": TOLERANCE, "near_zero_ratio": NEAR_ZERO_RATIO,
              "reference": "NumPy W @ x; no global phase alignment",
              "scale_convention": "s=max(1,sigma_max); t=sigma/s; y_optical=(W/s)x; y=s*y_optical",
              "special_cases": [name for name, _ in special_matrices(2)],
              "sizes": rows, "passed": all(r["passed"] for r in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                           + "\n", encoding="utf-8")
    print("报告:", args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
