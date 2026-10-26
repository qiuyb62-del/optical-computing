"""Clements 矩形网格分解，直接使用本项目的 MZI 约定。

数学来源：Clements et al., Optica 3, 1460–1465 (2016).
https://doi.org/10.1364/OPTICA.3.001460
实现采用交替左右消元，再解析地把左侧逆变换移过对角相位。
推导和传播顺序见 docs/clements-decomposition.md。
"""
import numpy as np
from .mzi import mzi_matrix

UNITARY_TOL = 1e-10


def _wrap_phase(phase):
    """映射到 [0,2π)，处理极小负数取模后被舍入为 2π 的情况。"""
    period = 2 * np.pi
    wrapped = np.remainder(phase, period)
    return np.where(wrapped >= period, 0.0, wrapped)


def _validate_unitary(U):
    try:
        matrix = np.array(U, dtype=np.complex128, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("U must be a finite square complex matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("U must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("U must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        error = np.linalg.norm(matrix.conj().T @ matrix - np.eye(len(matrix)), "fro")
    if not np.isfinite(error) or error > UNITARY_TOL:
        raise ValueError(f"U is not unitary: ||U†U-I||_F={error:.3e}")
    return matrix


def _null_angles(target, pivot):
    """让 [target,pivot] @ T† 的第一项为零；不用复数相除。

    atan2(0,0)=0；恰为零的元素不要求相位，避免 0/0。
    极小但非零的元素仍保留，不以人为阈值截断。
    """
    theta = float(np.arctan2(abs(target), abs(pivot)))
    phi = 0.0 if target == 0 or pivot == 0 else float(np.angle(target) - np.angle(pivot))
    return theta, phi


def decompose_clements(U):
    """返回 (ops, output_phases)，使 U ≈ D T_k ... T_1。

    ops: 按实际传播顺序的 (m, theta, phi)，作用于 (m,m+1)。
    保留完整 N(N-1)/2 个单元，包括零混合角单元。
    theta 为有效混合角，可为负；phi 和 output_phases ∈ [0,2π)。
    不修改 U，不把非酉输入自动投影为酉矩阵；支持 N=1。
    """
    reduced = _validate_unitary(U)
    n = len(reduced)
    ops, left_ops = [], []
    # 沿反对角线交替右消元和左消元，构造矩形网格。
    for diagonal in range(n - 1):
        for offset in range(diagonal + 1):
            if diagonal % 2 == 0:
                row, m = n - 1 - offset, diagonal - offset
                theta, phi = _null_angles(reduced[row, m], reduced[row, m + 1])
                reduced[:, m:m + 2] = reduced[:, m:m + 2] @ mzi_matrix(theta, phi).conj().T
                ops.append((m, theta, float(_wrap_phase(phi))))
            else:
                m, col = n + offset - diagonal - 2, offset
                theta, phi = _null_angles(-reduced[m + 1, col], reduced[m, col])
                reduced[m:m + 2, :] = mzi_matrix(theta, phi) @ reduced[m:m + 2, :]
                left_ops.append((m, theta, phi))
    phases = np.angle(np.diag(reduced)).copy()
    # 精确恒等式 T(θ,φ)† diag(e^ia,e^ib)
    # = diag(e^i(b-φ),e^ib) T(-θ,a-b)。从最后一次左消元开始搬移。
    for m, theta, phi in reversed(left_ops):
        a, b = phases[m], phases[m + 1]
        ops.append((m, -theta, float(_wrap_phase(a - b))))
        phases[m] = b - phi
    return ops, np.asarray(_wrap_phase(phases), dtype=np.float64)
