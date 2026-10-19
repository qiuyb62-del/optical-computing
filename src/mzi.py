"""固定约定：T = [[exp(iφ)cosθ, -sinθ], [exp(iφ)sinθ, cosθ]]。

角度为 rad；传播复数光场而非功率；端口顺序为上、下。
θ 是有效混合角，不直接等同于两臂物理相位差。
"""
from numbers import Real, Integral
import numpy as np


def _angle(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar in radians")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _field(E):
    try:
        result = np.asarray(E, dtype=np.complex128)
    except (ValueError, TypeError) as exc:
        raise ValueError("E must be a complex vector") from exc
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError("E must be a finite one-dimensional complex vector")
    return result


def mzi_matrix(theta, phi):
    """返回 (2,2) complex128 矩阵，不对 θ 按 π 取模。"""
    theta, phi = _angle(theta, "theta"), _angle(phi, "phi")
    c, s, phase = np.cos(theta), np.sin(theta), np.exp(1j * phi)
    return np.array([[phase * c, -s], [phase * s, c]], dtype=np.complex128)


def mzi(E1, E2, theta, phi):
    """两个复数标量光场输入，返回 (上路输出, 下路输出)。"""
    fields = _field([E1, E2])
    output = mzi_matrix(theta, phi) @ fields
    return output[0], output[1]


def apply_mzi(E, m, theta, phi):
    """返回新数组，只更新相邻通道 (m,m+1)，不修改原输入。"""
    fields = _field(E)
    if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
        raise ValueError("m must be an integer channel index")
    if not 0 <= m < len(fields) - 1:
        raise ValueError("m and m+1 must be valid channels")
    output = fields.copy()
    output[m:m+2] = mzi_matrix(theta, phi) @ fields[m:m+2]
    return output
