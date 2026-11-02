"""一般方阵 W 的理想光路：V† 网格 → 幅度衰减 → U 网格。

光学部分实现 W / recovery_scale；恢复比例是输出端的数字操作。
只在 compile_svd 中进行 SVD；传播复用参数，不调用 W @ x 或稠密 SVD 因子。
"""
from dataclasses import dataclass
import numpy as np
from .clements import decompose_clements
from .mesh import propagate_mesh, reconstruct_mesh
from .mzi import _field


@dataclass(frozen=True)
class SVDCircuit:
    """由 compile_svd 创建的只读配置；所有角度 rad，衰减系数作用于幅度。"""
    input_ops: tuple
    input_phases: tuple
    singular_values: tuple
    amplitude_transmissions: tuple
    recovery_scale: float
    output_ops: tuple
    output_phases: tuple

    @property
    def size(self):
        return len(self.amplitude_transmissions)

    def to_dict(self):
        """可写入 JSON 的全部器件参数；不保存目标矩阵或稠密 SVD 因子。"""
        return {
            "size": self.size,
            "input_mesh": {"ops": self.input_ops, "output_phases": self.input_phases},
            "singular_values": self.singular_values,
            "amplitude_transmissions": self.amplitude_transmissions,
            "recovery_scale": self.recovery_scale,
            "output_mesh": {"ops": self.output_ops, "output_phases": self.output_phases},
        }


@dataclass(frozen=True)
class SVDPropagation:
    """每一步的独立光场数组；restored_output 为数字恢复后的数值。"""
    after_input_mesh: np.ndarray
    after_attenuation: np.ndarray
    optical_output: np.ndarray
    restored_output: np.ndarray


def _square_matrix(W):
    try:
        matrix = np.array(W, dtype=np.complex128, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("W must be a finite nonempty square matrix") from exc
    if (matrix.ndim != 2 or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1]):
        raise ValueError("W must be a nonempty square matrix; rectangular matrices are not supported")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("W must contain only finite values")
    return matrix


def compile_svd(W):
    """将实数或复数 N×N 矩阵编译成两套 Clements 参数及被动衰减系数。

    NumPy 返回的 Vh 已是 V†，不能再次转置。s=max(1, sigma_max)，
    t_i=sigma_i/s ∈ [0,1]。零矩阵 s=1、t=0；不截断小奇异值。
    每套网格保留 N(N-1)/2 个 MZI，输入 W 不变。
    """
    matrix = _square_matrix(W)
    try:
        U, sigma, Vh = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("SVD did not converge for W") from exc
    if not np.all(np.isfinite(sigma)):
        raise ValueError("singular values exceed the finite float64 range; rescale W")
    scale = max(1.0, float(sigma[0]))
    input_ops, input_phases = decompose_clements(Vh)
    output_ops, output_phases = decompose_clements(U)
    return SVDCircuit(
        tuple(input_ops), tuple(map(float, input_phases)), tuple(map(float, sigma)),
        tuple(map(float, sigma / scale)), scale,
        tuple(output_ops), tuple(map(float, output_phases)),
    )


def trace_svd(E, circuit):
    """逐器件传播并返回各阶段光场；不会重新分解矩阵或归一化输入。"""
    if not isinstance(circuit, SVDCircuit):
        raise ValueError("circuit must be an SVDCircuit returned by compile_svd")
    fields = _field(E)
    if fields.shape != (circuit.size,):
        raise ValueError("E length must match the circuit size")
    try:
        with np.errstate(over="raise", invalid="raise"):
            first = propagate_mesh(fields, circuit.input_ops, circuit.input_phases)
            attenuated = np.asarray(circuit.amplitude_transmissions) * first
            optical = propagate_mesh(attenuated, circuit.output_ops, circuit.output_phases)
            restored = circuit.recovery_scale * optical
    except FloatingPointError as exc:
        raise ValueError("propagation exceeded the finite complex128 range; rescale W or E") from exc
    return SVDPropagation(first, attenuated, optical, restored)


def propagate_svd(E, circuit, *, restore_scale=True):
    """默认返回 y≈W@E；restore_scale=False 返回实际光路输出≈(W/s)@E。"""
    result = trace_svd(E, circuit)
    return result.restored_output if restore_scale else result.optical_output


def reconstruct_svd(circuit, *, restore_scale=True):
    """仅用于检查：从器件参数重建 W 或 W/s；传播不使用本函数。"""
    if not isinstance(circuit, SVDCircuit):
        raise ValueError("circuit must be an SVDCircuit returned by compile_svd")
    left = reconstruct_mesh(circuit.output_ops, circuit.output_phases)
    right = reconstruct_mesh(circuit.input_ops, circuit.input_phases)
    optical = (left * np.asarray(circuit.amplitude_transmissions)[None, :]) @ right
    return circuit.recovery_scale * optical if restore_scale else optical
