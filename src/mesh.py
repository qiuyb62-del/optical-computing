"""相邻通道 MZI 网格；ops 按传播顺序，输出端相位最后作用。"""
from numbers import Integral
import numpy as np
from .mzi import _angle, _field, mzi_matrix


def _validate_mesh(ops, output_phases):
    """规范化参数并拒绝形状、复相位、越界和非有限值。"""
    try:
        raw = np.asarray(output_phases)
        if raw.ndim != 1 or raw.size == 0 or np.iscomplexobj(raw):
            raise ValueError("output_phases must be a nonempty real vector")
        phases = np.asarray([_angle(x, "output phase") for x in raw], dtype=float)
        operations = list(ops)
    except (TypeError, OverflowError) as exc:
        raise ValueError("invalid mesh parameters") from exc
    normalized = []
    for op in operations:
        try:
            m, theta, phi = op
        except (TypeError, ValueError) as exc:
            raise ValueError("each operation must be (m, theta, phi)") from exc
        if isinstance(m, (bool, np.bool_)) or not isinstance(m, Integral):
            raise ValueError("m must be an integer channel index")
        if not 0 <= m < phases.size - 1:
            raise ValueError("operation must address adjacent valid channels")
        normalized.append((int(m), _angle(theta, "theta"), _angle(phi, "phi")))
    return normalized, phases


def propagate_mesh(E, ops, output_phases):
    """逐个应用 MZI 后乘输出端相位；返回新 complex128 向量。

    此接口不预先生成整个矩阵，用来模拟光场依次经过器件。
    """
    operations, phases = _validate_mesh(ops, output_phases)
    fields = _field(E)
    if fields.shape != phases.shape:
        raise ValueError("E length must match output_phases length")
    result = fields.copy()
    for m, theta, phi in operations:
        result[m:m + 2] = mzi_matrix(theta, phi) @ result[m:m + 2]
    return np.asarray(np.exp(1j * phases) * result, dtype=np.complex128)


def reconstruct_mesh(ops, output_phases):
    """返回 Uhat = D T_k ... T_1；T_j 为嵌入 N 通道的 MZI。"""
    operations, phases = _validate_mesh(ops, output_phases)
    matrix = np.eye(phases.size, dtype=np.complex128)
    for m, theta, phi in operations:
        matrix[m:m + 2, :] = mzi_matrix(theta, phi) @ matrix[m:m + 2, :]
    return np.exp(1j * phases)[:, None] * matrix


def mesh_layers(ops, output_phases):
    """把不共享通道的单元排到同一层，保留每条通道上的先后依赖。

    返回各层的 operation 列表；D 不在层内，始终最后作用。
    可用于展示矩形布局，不改变 ops 的数学意义。
    """
    operations, phases = _validate_mesh(ops, output_phases)
    last_layer = [-1] * phases.size
    layers = []
    for op in operations:
        m = op[0]
        level = max(last_layer[m], last_layer[m + 1]) + 1
        while len(layers) <= level:
            layers.append([])
        layers[level].append(op)
        last_layer[m] = last_layer[m + 1] = level
    return layers
