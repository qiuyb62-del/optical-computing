"""第三阶段示例与批量报告共用的误差定义；不参与光路计算。"""
import numpy as np

TOLERANCE = 1e-10
NEAR_ZERO_RATIO = 1e-12


def complex_data(value):
    value = np.asarray(value)
    return {"real": value.real.tolist(), "imag": value.imag.tolist()}


def comparison(actual, expected, *, reference_scale=0.0):
    """普通结果检查相对误差；零或近零结果检查有尺度的绝对误差。

    reference_scale 为 ||W||_F ||x||_2（矩阵重建不用此参数）。
    近零指 ||expected|| <= 1e-12 * reference_scale，不按输出自身放大噪声。
    相对误差始终如实记录；零分母记 None，JSON 中为 null。
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError("comparison shapes must match")
    error = float(np.linalg.norm(actual - expected))
    norm = float(np.linalg.norm(expected))
    relative = error / norm if norm > 0 else None
    near_zero = norm == 0 or norm <= NEAR_ZERO_RATIO * reference_scale
    mode = "absolute_near_zero" if near_zero else "relative"
    limit = TOLERANCE * max(1.0, reference_scale) if near_zero else TOLERANCE * norm
    finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
                  and np.isfinite(error) and np.isfinite(norm) and np.isfinite(limit))
    return {"absolute_error": error, "reference_norm": norm,
            "relative_error": relative, "criterion": mode, "absolute_limit": limit,
            "passed": finite and error <= limit}


def evaluate(W, x, circuit, trace, reconstructed):
    """分别检查矩阵、恢复输出、原始光场、两套网格功率与被动性。"""
    reference = W @ x
    scale = float(np.linalg.norm(W, "fro") * np.linalg.norm(x))
    matrix = comparison(reconstructed, W)
    vector = comparison(trace.restored_output, reference, reference_scale=scale)
    optical = comparison(trace.optical_output, reference / circuit.recovery_scale,
                         reference_scale=scale / circuit.recovery_scale)
    values = [x, trace.after_input_mesh, trace.after_attenuation, trace.optical_output]
    powers = [float(np.vdot(v, v).real) for v in values]
    p0, p1, p2, p3 = powers
    # 非零输入以输入功率为尺度；零输入要求各阶段严格为零。
    limit = TOLERANCE * p0
    power_passed = bool(np.all(np.isfinite(powers))
                        and abs(p1 - p0) <= limit and abs(p3 - p2) <= limit
                        and p2 <= p1 + limit and p3 <= p0 + limit)
    return {"matrix": matrix, "vector": vector, "optical_vector": optical,
            "powers": dict(zip(["input", "after_input_mesh", "after_attenuation",
                                 "optical_output"], powers)),
            "power_checks_passed": power_passed,
            "passed": matrix["passed"] and vector["passed"] and optical["passed"] and power_passed}
