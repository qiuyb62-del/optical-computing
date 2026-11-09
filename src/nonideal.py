"""第四阶段：有固定器件误差的被动光路，逐个 2×2 单元传播。

量化作用于显式物理相位控制量，噪声在量化后加入；不直接量化有效 θ。
realize_nonideal 采样一次静态器件，后续传播不再产生随机数。
这仍是传输矩阵模型，不是 Lumerical 或空间电磁场仿真。
"""
from dataclasses import asdict, dataclass
from numbers import Integral
import numpy as np
from .mzi import _angle, _field
from .mesh import _validate_mesh
from .svd_matmul import SVDCircuit, SVDPropagation

TAU = 2.0 * np.pi


def _nonnegative(value, name):
    value = _angle(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _bits(value):
    if value is None:
        return None
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or not 1 <= value <= 32):
        raise ValueError("phase_bits must be None or an integer from 1 to 32")
    return int(value)


def _seed(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < 0):
        raise ValueError("seed must be a nonnegative integer")
    return int(value)


def _coupling(value):
    value = _angle(value, "coupler_through")
    if not 0 <= value <= 1:
        raise ValueError("coupler_through must be a power fraction in [0, 1]")
    return value


@dataclass(frozen=True)
class NonidealConfig:
    """器件控制误差：所有损耗以功率 dB 描述，phase_std_rad 以 rad 描述。

    phase_std_rad: 每个 MZI 的四个相位控制及每个网格端相位的独立静态偏差。
    loss_db_per_mzi: 每个 MZI 两输出相同的额外插入损耗。
    upper_arm_loss_db: 每个 MZI 内部上臂的额外损耗，单独模拟不平衡。
    coupler_through: 两个耦合器相同的直通功率比，理想为 0.5。
    """
    phase_std_rad: float = 0.0
    loss_db_per_mzi: float = 0.0
    phase_bits: int | None = None
    coupler_through: float = 0.5
    upper_arm_loss_db: float = 0.0

    def __post_init__(self):
        for key in ("phase_std_rad", "loss_db_per_mzi", "upper_arm_loss_db"):
            object.__setattr__(self, key, _nonnegative(getattr(self, key), key))
        object.__setattr__(self, "phase_bits", _bits(self.phase_bits))
        object.__setattr__(self, "coupler_through", _coupling(self.coupler_through))


def quantize_phase(phase, bits=None):
    """[0,2π) 周期相位量化；最近格点；恰好中点采用 round-to-even。

    bits=None 保留原相位，不折叠有效混合角；所有调用必须传实数标量。
    """
    phase, bits = _angle(phase, "phase"), _bits(bits)
    if bits is None:
        return phase
    levels = 2 ** bits
    code = int(np.rint((phase % TAU) / TAU * levels)) % levels
    return float(code * TAU / levels)


def coupler_matrix(through=0.5):
    """无损互易耦合器 B；through 是功率比，不是场幅度比。"""
    through = _coupling(through)
    a, b = np.sqrt(through), 1j * np.sqrt(1.0 - through)
    return np.array([[a, b], [b, a]], dtype=np.complex128)


def physical_mzi_matrix(phases, *, coupler_through=0.5,
                        loss_db=0.0, upper_arm_loss_db=0.0):
    """相位顺序 (φ, δ, χ_upper, χ_lower)，输出耦合器采用 B† 端口约定。

    Dχ B† diag(exp(iδ)*a_arm,1) B diag(exp(iφ),1) * a_loss。
    理想映射 δ=2θ, χ_upper=χ_lower=−θ，严格得到固定 T(θ,φ)。
    补偿相位不可省略，否则会产生依赖 θ 的公共相位。
    """
    try:
        phi, delta, chi_upper, chi_lower = [_angle(v, "phase") for v in phases]
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must contain four finite real phase controls") from exc
    loss_db = _nonnegative(loss_db, "loss_db")
    upper_arm_loss_db = _nonnegative(upper_arm_loss_db, "upper_arm_loss_db")
    B = coupler_matrix(coupler_through)
    P = np.diag([np.exp(1j * phi), 1.0])
    arm = np.diag([np.exp(1j * delta) * 10.0 ** (-upper_arm_loss_db / 20.0), 1.0])
    output = np.diag(np.exp(1j * np.array([chi_upper, chi_lower])))
    return 10.0 ** (-loss_db / 20.0) * (output @ B.conj().T @ arm @ B @ P)


@dataclass(frozen=True)
class RealizedGate:
    channel: int
    commanded_phases: tuple
    actual_phases: tuple
    matrix: tuple


@dataclass(frozen=True)
class RealizedMesh:
    gates: tuple
    commanded_output_phases: tuple
    actual_output_phases: tuple


@dataclass(frozen=True)
class NonidealCircuit:
    input_mesh: RealizedMesh
    output_mesh: RealizedMesh
    amplitude_transmissions: tuple
    recovery_scale: float
    config: NonidealConfig
    seed: int

    @property
    def size(self):
        return len(self.amplitude_transmissions)

    def to_dict(self):
        """完整静态器件快照；局部矩阵用实部/虚部保存，不保存稠密 W。"""
        def mesh_data(mesh):
            gates = []
            for gate in mesh.gates:
                m = np.asarray(gate.matrix)
                gates.append({"channel": gate.channel,
                              "commanded_phases": gate.commanded_phases,
                              "actual_phases": gate.actual_phases,
                              "matrix": {"real": m.real.tolist(), "imag": m.imag.tolist()}})
            return {"gates": gates,
                    "commanded_output_phases": mesh.commanded_output_phases,
                    "actual_output_phases": mesh.actual_output_phases}
        return {"model": "passive_physical_phase_v1", "size": self.size,
                "noise_mode": "static_once_per_realization", "seed": self.seed,
                "config": asdict(self.config),
                "control_order": ["phi", "delta", "chi_upper", "chi_lower"],
                "input_mesh": mesh_data(self.input_mesh),
                "output_mesh": mesh_data(self.output_mesh),
                "amplitude_transmissions": self.amplitude_transmissions,
                "recovery_scale": self.recovery_scale}


def realize_nonideal(circuit, config=None, *, seed=0):
    """由理想配置生成一台固定的非理想器件；相位偏差在量化之后采样。

    相同配置和 seed 可复现；同一器件反复输入不同向量不会重新抽样。
    为配对比较，即使 σ=0 也抽取同样顺序的标准正态随机数。
    """
    if not isinstance(circuit, SVDCircuit):
        raise ValueError("circuit must be an SVDCircuit returned by compile_svd")
    if config is None:
        config = NonidealConfig()
    if not isinstance(config, NonidealConfig):
        raise ValueError("config must be a NonidealConfig")
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    transmissions = tuple(_nonnegative(t, "transmission") for t in circuit.amplitude_transmissions)
    if not transmissions or any(t > 1 for t in transmissions):
        raise ValueError("amplitude transmissions must be a nonempty sequence in [0,1]")
    scale = _angle(circuit.recovery_scale, "recovery_scale")
    if scale < 1:
        raise ValueError("recovery_scale must be >= 1")

    def controls(values):
        command = tuple(quantize_phase(v, config.phase_bits) for v in values)
        offsets = rng.normal(size=len(command)) * config.phase_std_rad
        with np.errstate(over="ignore", invalid="ignore"):
            actual = np.asarray(command) + offsets
        if not np.all(np.isfinite(actual)):
            raise ValueError("phase settings exceed the finite float64 range")
        return command, tuple(map(float, actual))

    def realize_mesh(ops, output_phases):
        operations, phases = _validate_mesh(ops, output_phases)
        if len(phases) != len(transmissions):
            raise ValueError("mesh size must match transmissions")
        gates = []
        for m, theta, phi in operations:
            # 先按物理控制的 2π 周期约化，避免 2θ 溢出；不按 π 改写 θ。
            theta_wrapped = theta % TAU
            command, actual = controls((phi % TAU, (2 * theta_wrapped) % TAU,
                                        (-theta_wrapped) % TAU, (-theta_wrapped) % TAU))
            matrix = physical_mzi_matrix(actual, coupler_through=config.coupler_through,
                                         loss_db=config.loss_db_per_mzi,
                                         upper_arm_loss_db=config.upper_arm_loss_db)
            gates.append(RealizedGate(m, command, actual,
                                      tuple(tuple(complex(v) for v in row) for row in matrix)))
        command, actual = controls(tuple(float(v % TAU) for v in phases))
        return RealizedMesh(tuple(gates), command, actual)

    first = realize_mesh(circuit.input_ops, circuit.input_phases)
    last = realize_mesh(circuit.output_ops, circuit.output_phases)
    return NonidealCircuit(first, last, transmissions, scale, config, seed)


def _propagate_mesh(fields, mesh):
    result = fields.copy()
    for gate in mesh.gates:
        m = gate.channel
        result[m:m + 2] = np.asarray(gate.matrix) @ result[m:m + 2]
    return np.exp(1j * np.asarray(mesh.actual_output_phases)) * result


def trace_nonideal(E, circuit):
    """与第三阶段相同的四个观察点；传播本身无随机性，不修改输入。"""
    if not isinstance(circuit, NonidealCircuit):
        raise ValueError("circuit must be returned by realize_nonideal")
    fields = _field(E)
    if fields.shape != (circuit.size,):
        raise ValueError("E length must match circuit size")
    try:
        with np.errstate(over="raise", invalid="raise"):
            first = _propagate_mesh(fields, circuit.input_mesh)
            attenuated = np.asarray(circuit.amplitude_transmissions) * first
            optical = _propagate_mesh(attenuated, circuit.output_mesh)
            restored = circuit.recovery_scale * optical
    except FloatingPointError as exc:
        raise ValueError("propagation overflow; rescale W or E") from exc
    return SVDPropagation(first, attenuated, optical, restored)


def propagate_nonideal(E, circuit, *, restore_scale=True):
    trace = trace_nonideal(E, circuit)
    return trace.restored_output if restore_scale else trace.optical_output


def reconstruct_nonideal(circuit, *, restore_scale=True):
    """仅用于验收：逐个基向量恢复整体传输矩阵；正常传播不调用此函数。"""
    if not isinstance(circuit, NonidealCircuit):
        raise ValueError("circuit must be returned by realize_nonideal")
    return np.column_stack([propagate_nonideal(e, circuit, restore_scale=restore_scale)
                            for e in np.eye(circuit.size, dtype=np.complex128)])
