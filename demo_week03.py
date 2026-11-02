"""第三周：一般矩阵 W → 两套 MZI 网格与衰减器 → 输出 → NumPy 对照。"""
import argparse
import json
from pathlib import Path
import numpy as np
from src.svd_matmul import compile_svd, trace_svd, reconstruct_svd
from src.svd_validation import complex_data, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4, help="方阵尺寸，默认 4")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--real", action="store_true", help="生成实数 W 和 x；默认复数")
    parser.add_argument("--report", type=Path, help="保存完整参数、中间光场和误差为 JSON")
    args = parser.parse_args()
    if args.size < 1 or args.seed < 0:
        parser.error("size 必须为正整数；seed 必须为非负整数")
    rng = np.random.default_rng(args.seed)
    W = rng.normal(size=(args.size, args.size)).astype(np.complex128)
    x = rng.normal(size=args.size).astype(np.complex128)
    if not args.real:
        W += 1j * rng.normal(size=W.shape)
        x += 1j * rng.normal(size=x.shape)
    circuit = compile_svd(W)
    trace = trace_svd(x, circuit)
    metrics = evaluate(W, x, circuit, trace, reconstruct_svd(circuit))
    np.set_printoptions(precision=5, suppress=True)
    print(f"SVD 矩阵乘法：{args.size} 通道，seed={args.seed}")
    print("目标矩阵 W:\n", W)
    print("输入复数光场 x:", x)
    print("奇异值 sigma:", np.asarray(circuit.singular_values))
    print(f"数字恢复比例 s={circuit.recovery_scale:.8f}")
    print("各通道幅度透过率 t=sigma/s:", np.asarray(circuit.amplitude_transmissions))
    print(f"MZI 数量：输入网格 {len(circuit.input_ops)}，输出网格 {len(circuit.output_ops)}")
    print("1. V† 网格后: ", trace.after_input_mesh)
    print("2. 幅度衰减后:", trace.after_attenuation)
    print("3. U 网格后（光路输出，目标为 W@x/s）:", trace.optical_output)
    print("4. 数字乘 s 后（计算结果）:", trace.restored_output)
    print("NumPy 直接计算 W @ x:      ", W @ x)
    for name, label in [("matrix", "矩阵重建"), ("vector", "恢复后输出")]:
        metric = metrics[name]
        if metric["criterion"] == "relative":
            print(f"{label}相对误差: {metric['relative_error']:.3e}")
        else:
            print(f"{label}绝对误差: {metric['absolute_error']:.3e}（零/近零输出）")
    print("各阶段总功率:", metrics["powers"])
    print("结果:", "PASS" if metrics["passed"] else "FAIL")
    if args.report:
        report = {"size": args.size, "seed": args.seed, "real_input": args.real,
                  "W": complex_data(W), "input": complex_data(x), "circuit": circuit.to_dict(),
                  "after_input_mesh": complex_data(trace.after_input_mesh),
                  "after_attenuation": complex_data(trace.after_attenuation),
                  "optical_output": complex_data(trace.optical_output),
                  "restored_output": complex_data(trace.restored_output),
                  "digital_output": complex_data(W @ x), "metrics": metrics}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                               + "\n", encoding="utf-8")
        print("已保存:", args.report)
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
