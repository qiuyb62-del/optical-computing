"""第二周：目标酉矩阵 → Clements 参数 → 逐器件传播 → 数字对照。"""
import argparse
import json
from pathlib import Path
import numpy as np
from src.clements import decompose_clements
from src.mesh import mesh_layers, propagate_mesh, reconstruct_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4, help="通道数，默认 4")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--report", type=Path, help="可选：保存参数和复数输入输出为 JSON")
    args = parser.parse_args()
    if args.size < 1 or args.seed < 0:
        parser.error("size 必须为正整数；seed 必须为非负整数")
    rng = np.random.default_rng(args.seed)
    n = args.size
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    U, r = np.linalg.qr(z)
    d = np.diag(r)
    U = U * (d / np.abs(d))[None, :]
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    ops, phases = decompose_clements(U)
    Uhat = reconstruct_mesh(ops, phases)
    y = propagate_mesh(x, ops, phases)
    expected = U @ x
    matrix_error = float(np.linalg.norm(Uhat - U, "fro") / np.linalg.norm(U, "fro"))
    vector_error = float(np.linalg.norm(y - expected) / np.linalg.norm(expected))
    power_in, power_out = float(np.vdot(x, x).real), float(np.vdot(y, y).real)
    np.set_printoptions(precision=5, suppress=True)
    print(f"Clements 示例：{n} 通道，seed={args.seed}，{len(ops)} 个 MZI")
    print("目标矩阵 U:\n", U)
    print("输入光场 x:", x)
    print("实际传播顺序（通道从 0 编号，角度为 rad）：")
    for step, (m, theta, phi) in enumerate(ops, 1):
        print(f"  {step:2d}. 通道 ({m},{m + 1})  theta={theta:.8f}  phi={phi:.8f}")
    print("矩形网格各层通道对：", [[(m, m + 1) for m, _, _ in layer]
                                    for layer in mesh_layers(ops, phases)])
    print("最后施加的输出端相位:", phases)
    print("网格逐器件输出: ", y)
    print("数字矩阵 U @ x:", expected)
    print(f"矩阵重建相对误差: {matrix_error:.3e}")
    print(f"光场输出相对误差: {vector_error:.3e}")
    print(f"输入总功率: {power_in:.12f}；输出总功率: {power_out:.12f}")
    passed = matrix_error <= 1e-10 and vector_error <= 1e-10
    print("结果:", "PASS" if passed else "FAIL")
    if args.report:
        def complex_data(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        report = {"size": n, "seed": args.seed, "U": complex_data(U),
                  "input": complex_data(x), "ops": ops, "output_phases": phases.tolist(),
                  "optical_output": complex_data(y), "digital_output": complex_data(expected),
                  "matrix_relative_error": matrix_error, "vector_relative_error": vector_error,
                  "input_power": power_in, "output_power": power_out, "passed": passed}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("已保存:", args.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
