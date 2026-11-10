"""Build and measure passive Clements/SVD circuits in Lumerical INTERCONNECT.

The module intentionally uses only local primitive elements: Waveguide Coupler,
Optical Phase Shift and Optical Attenuator.  A matrix is never inserted as a
black box; every 2x2 gate is assembled from two couplers and four phase
shifters, including the two output reference phases of the project's physical
MZI model.

Measured port/phase conventions (Lumerical 2025 R2, build 4175):

* ``Waveguide Coupler`` with ``conjugate=0`` has, from left (1,2) to right
  (3,4), ``B = [[1, i], [i, 1]] / sqrt(2)``.
* ``conjugate=1`` changes the coupling sign and gives ``B†``.  This is a fixed
  reference-plane choice, not an assumption about a generic coupler.
* ``Optical Phase Shift`` applies ``exp(+i phase)``.
* Therefore the project MZI is: input P(phi) -> B -> upper-arm P(2 theta) ->
  B† -> output P(-theta) on both ports.  This was checked against T(theta,phi)
  for theta=0, pi/4, pi/2 and several phi values before this builder was used.

Angles are radians, fields are complex, channels are numbered from zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


DEFAULT_LUMERICAL_ROOT = Path(r"D:\Program Files\ANSYS Inc\v252\Lumerical")
LUMERICAL_ROOT = Path(os.environ.get("LUMERICAL_ROOT", DEFAULT_LUMERICAL_ROOT))
LUMERICAL_API = Path(os.environ.get("LUMERICAL_API", LUMERICAL_ROOT / "api" / "python"))
LUMERICAL_PYTHON = Path(os.environ.get(
    "LUMERICAL_PYTHON", LUMERICAL_ROOT / "python-3.13.1" / "python.exe"))


def resolve_lumerical_paths(root=None):
    """Resolve install/API paths, allowing env vars or an explicit root.

    Priority: explicit ``root`` > ``LUMERICAL_API`` > ``LUMERICAL_ROOT`` >
    the detected local default.  The caller can therefore use another
    machine's Lumerical without editing source code.
    """
    if root is not None:
        install = Path(root).expanduser().resolve()
        return install, install / "api" / "python", install / "python-3.13.1" / "python.exe"
    env_api = os.environ.get("LUMERICAL_API")
    if env_api:
        api = Path(env_api).expanduser().resolve()
        install = api.parents[1] if len(api.parents) >= 2 else api.parent
        python = install / "python-3.13.1" / "python.exe"
        return install, api, python
    install = Path(os.environ.get("LUMERICAL_ROOT", DEFAULT_LUMERICAL_ROOT)).expanduser().resolve()
    return install, install / "api" / "python", install / "python-3.13.1" / "python.exe"


def load_lumapi(root=None):
    """Import the official lumapi from an explicit, env, or default install."""
    _, api, _ = resolve_lumerical_paths(root)
    api_file = api / "lumapi.py"
    if not api_file.is_file():
        raise RuntimeError(f"Lumerical API not found: {api_file}")
    if str(api) not in sys.path:
        sys.path.insert(0, str(api))
    import lumapi  # noqa: PLC0415 - deliberately delayed until the path is set

    return lumapi


def app_version(root=None) -> str:
    install, _, _ = resolve_lumerical_paths(root)
    version = install / "VERSION"
    return version.read_text(encoding="utf-8").strip() if version.exists() else "unknown"


@dataclass
class Port:
    element: str
    port: str

    def as_tuple(self):
        return (self.element, self.port)


def _name(app, fallback):
    try:
        return str(app.get("name"))
    except Exception:
        return fallback


def add(app, element: str, name: str | None = None) -> str:
    """Add one element and give it a deterministic, unique name when possible."""
    app.addelement(element)
    actual = _name(app, name or element)
    if name is not None and actual != name:
        app.set("name", name)
        actual = _name(app, name)
    return actual


def phase_shifter(app, name: str, phase: float = 0.0) -> str:
    element = add(app, "Optical Phase Shift", name)
    app.setnamed(element, "phase shift", float(phase))
    return element


def coupler(app, name: str, *, conjugate: int = 0, through: float = 0.5) -> str:
    """Waveguide coupler; ``through`` is the straight-through **power** fraction.

    Lumerical's ``coupling coefficient`` is the cross-coupled power: measured
    coefficients 0.49 and 0.51 give output amplitudes
    ``[sqrt(0.51), i sqrt(0.49)]``.  The project's ``coupler_through=eta`` is
    the straight-through power, so the Lumerical setting is ``1-eta``.
    """
    through = float(through)
    if not 0.0 <= through <= 1.0:
        raise ValueError("coupler through must be a power fraction in [0,1]")
    element = add(app, "Waveguide Coupler", name)
    app.setnamed(element, "conjugate", int(conjugate))
    app.setnamed(element, "coupling coefficient 1", 1.0 - through)
    app.setnamed(element, "coupling coefficient 2", 1.0 - through)
    app.setnamed(element, "insertion loss", 0.0)
    return element


def attenuator(app, name: str, amplitude: float) -> str:
    """Amplitude transmission in [0, 1]; Lumerical's attenuation is power dB."""
    amplitude = float(amplitude)
    if not (0.0 <= amplitude <= 1.0):
        raise ValueError("attenuator amplitude must be in [0,1]")
    element = add(app, "Optical Attenuator", name)
    power_db = 300.0 if amplitude == 0.0 else -20.0 * math.log10(amplitude)
    app.setnamed(element, "attenuation", float(power_db))
    return element


def connect(app, source: Port, target: Port) -> None:
    app.connect(source.element, source.port, target.element, target.port)


def disconnect(app, source: Port, target: Port) -> None:
    try:
        app.disconnect(source.element, source.port, target.element, target.port)
    except Exception:
        # A disconnected ONA source after a failed run should not mask the
        # original error; callers only use this for their own rewiring.
        pass


def add_mzi_phases(app, name: str, channel: int, phases, *, through: float = 0.5,
                   loss_db: float = 0.0, upper_arm_loss_db: float = 0.0):
    """Assemble one MZI from explicit physical controls (phi, delta, chi_u, chi_l).

    The optional loss attenuators reproduce the Python nonideal model: a
    uniform factor after the whole 2x2 MZI is placed on both outputs, while the
    upper-arm imbalance is placed between the two couplers.  ``loss_db`` values
    are power dB and are converted to amplitude transmission here.
    """
    phi, delta, chi_upper, chi_lower = [float(v) for v in phases]
    p_in = phase_shifter(app, f"{name}_PIN", phi)
    c1 = coupler(app, f"{name}_C1", conjugate=0, through=through)
    c2 = coupler(app, f"{name}_C2", conjugate=1, through=through)
    p_arm = phase_shifter(app, f"{name}_PARM", delta)
    p_up = phase_shifter(app, f"{name}_POUT_U", chi_upper)
    p_lo = phase_shifter(app, f"{name}_POUT_L", chi_lower)

    connect(app, Port(p_in, "port 2"), Port(c1, "port 1"))
    arm_input = Port(p_arm, "port 1")
    if upper_arm_loss_db:
        arm_att = attenuator(app, f"{name}_UARMLOSS", 10.0 ** (-float(upper_arm_loss_db) / 20.0))
        connect(app, Port(c1, "port 3"), Port(arm_att, "port 1"))
        connect(app, Port(arm_att, "port 2"), Port(p_arm, "port 1"))
    else:
        arm_att = None
        connect(app, Port(c1, "port 3"), arm_input)
    connect(app, Port(p_arm, "port 2"), Port(c2, "port 3"))
    connect(app, Port(c1, "port 4"), Port(c2, "port 4"))
    connect(app, Port(c2, "port 1"), Port(p_up, "port 1"))
    connect(app, Port(c2, "port 2"), Port(p_lo, "port 1"))

    output_ports = [Port(p_up, "port 2"), Port(p_lo, "port 2")]
    loss_atts = []
    if loss_db:
        amplitude = 10.0 ** (-float(loss_db) / 20.0)
        for j, port in enumerate(output_ports):
            loss_att = attenuator(app, f"{name}_LOSS{j}", amplitude)
            connect(app, port, Port(loss_att, "port 1"))
            output_ports[j] = Port(loss_att, "port 2")
            loss_atts.append(loss_att)
    return {
        "name": name,
        "channel": int(channel),
        "phases": (phi, delta, chi_upper, chi_lower),
        "inputs": [Port(p_in, "port 1"), Port(c1, "port 2")],
        "outputs": output_ports,
        "components": {
            "input_phase": p_in,
            "coupler_in": c1,
            "arm_phase": p_arm,
            "arm_loss": arm_att,
            "coupler_out": c2,
            "output_phase_upper": p_up,
            "output_phase_lower": p_lo,
            "output_attenuators": loss_atts,
        },
    }


def add_mzi(app, name: str, channel: int, theta: float, phi: float):
    """Assemble one ideal project MZI and return its input/output port descriptors."""
    gate = add_mzi_phases(
        app, name, channel,
        (float(phi), 2.0 * float(theta), -float(theta), -float(theta)),
    )
    gate.update({"theta": float(theta), "phi": float(phi)})
    return gate


def add_mesh(app, name: str, ops, output_phases):
    """Build one Clements mesh, including N input references and N output phases.

    ``ops`` must be the propagation-ordered ``(m, theta, phi)`` tuples from
    ``src.clements.decompose_clements``.  The returned ports are the external
    left inputs and right outputs; all internal connections are physical local
    elements.
    """
    n = len(output_phases)
    if n < 1:
        raise ValueError("mesh must have at least one channel")
    input_phases = [phase_shifter(app, f"{name}_IN{j}", 0.0) for j in range(n)]
    # Port 1 is the external mesh input; port 2 feeds the first gate.  Keeping
    # these separate prevents the first connection from consuming the exposed
    # input port (a bug caught by the 4-channel connection test).
    current: list[Port] = [Port(p, "port 2") for p in input_phases]
    gates = []
    for index, (m, theta, phi) in enumerate(ops):
        m = int(m)
        if not 0 <= m < n - 1:
            raise ValueError("MZI channel out of range")
        gate = add_mzi(app, f"{name}_M{index:04d}", m, theta, phi)
        connect(app, current[m], gate["inputs"][0])
        connect(app, current[m + 1], gate["inputs"][1])
        current[m] = gate["outputs"][0]
        current[m + 1] = gate["outputs"][1]
        gates.append({k: gate[k] for k in ("name", "channel", "theta", "phi")})
    output_ps = [phase_shifter(app, f"{name}_OUT{j}", output_phases[j]) for j in range(n)]
    for j in range(n):
        connect(app, current[j], Port(output_ps[j], "port 1"))
    return {
        "name": name,
        "size": n,
        "gates": gates,
        "inputs": [Port(p, "port 1") for p in input_phases],
        "outputs": [Port(p, "port 2") for p in output_ps],
        "output_phases": [float(x) for x in output_phases],
    }


def add_svd(app, name: str, circuit):
    """Build V† mesh -> amplitude attenuators -> U mesh from an SVDCircuit."""
    first = add_mesh(app, f"{name}_VH", circuit.input_ops, circuit.input_phases)
    attenuators = []
    for j, amplitude in enumerate(circuit.amplitude_transmissions):
        attenuators.append(attenuator(app, f"{name}_ATT{j}", amplitude))
    last = add_mesh(app, f"{name}_U", circuit.output_ops, circuit.output_phases)
    for j in range(circuit.size):
        connect(app, first["outputs"][j], Port(attenuators[j], "port 1"))
        connect(app, Port(attenuators[j], "port 2"), last["inputs"][j])
    return {
        "name": name,
        "size": circuit.size,
        "recovery_scale": float(circuit.recovery_scale),
        "amplitude_transmissions": [float(x) for x in circuit.amplitude_transmissions],
        "input_mesh": first,
        "output_mesh": last,
        "attenuators": attenuators,
        "inputs": first["inputs"],
        "outputs": last["outputs"],
    }


def add_mesh_nonideal(app, name: str, mesh, *, coupler_through: float = 0.5,
                      loss_db_per_mzi: float = 0.0, upper_arm_loss_db: float = 0.0):
    """Build a mesh from a realized Python ``RealizedMesh`` (actual phase values).

    The phase commands were quantized and perturbed once by
    ``src.nonideal.realize_nonideal``; this function only writes those fixed
    actual values into the Lumerical phase shifters, so different inputs reuse
    exactly the same device.
    """
    n = len(mesh.actual_output_phases)
    input_phases = [phase_shifter(app, f"{name}_IN{j}", 0.0) for j in range(n)]
    current: list[Port] = [Port(p, "port 2") for p in input_phases]
    gates = []
    for index, source_gate in enumerate(mesh.gates):
        m = int(source_gate.channel)
        gate = add_mzi_phases(
            app, f"{name}_M{index:04d}", m, source_gate.actual_phases,
            through=coupler_through, loss_db=loss_db_per_mzi,
            upper_arm_loss_db=upper_arm_loss_db,
        )
        connect(app, current[m], gate["inputs"][0])
        connect(app, current[m + 1], gate["inputs"][1])
        current[m] = gate["outputs"][0]
        current[m + 1] = gate["outputs"][1]
        gates.append({"name": gate["name"], "channel": m,
                      "actual_phases": [float(v) for v in source_gate.actual_phases]})
    output_ps = [phase_shifter(app, f"{name}_OUT{j}", mesh.actual_output_phases[j])
                 for j in range(n)]
    for j in range(n):
        connect(app, current[j], Port(output_ps[j], "port 1"))
    return {
        "name": name,
        "size": n,
        "gates": gates,
        "inputs": [Port(p, "port 1") for p in input_phases],
        "outputs": [Port(p, "port 2") for p in output_ps],
        "actual_output_phases": [float(x) for x in mesh.actual_output_phases],
    }


def add_svd_nonideal(app, name: str, device):
    """Build the full nonideal SVD circuit from a ``NonidealCircuit`` snapshot."""
    config = device.config
    first = add_mesh_nonideal(
        app, f"{name}_VH", device.input_mesh,
        coupler_through=config.coupler_through,
        loss_db_per_mzi=config.loss_db_per_mzi,
        upper_arm_loss_db=config.upper_arm_loss_db,
    )
    attenuators = [attenuator(app, f"{name}_ATT{j}", amplitude)
                   for j, amplitude in enumerate(device.amplitude_transmissions)]
    last = add_mesh_nonideal(
        app, f"{name}_U", device.output_mesh,
        coupler_through=config.coupler_through,
        loss_db_per_mzi=config.loss_db_per_mzi,
        upper_arm_loss_db=config.upper_arm_loss_db,
    )
    for j in range(device.size):
        connect(app, first["outputs"][j], Port(attenuators[j], "port 1"))
        connect(app, Port(attenuators[j], "port 2"), last["inputs"][j])
    return {
        "name": name,
        "size": device.size,
        "recovery_scale": float(device.recovery_scale),
        "amplitude_transmissions": [float(x) for x in device.amplitude_transmissions],
        "input_mesh": first,
        "output_mesh": last,
        "attenuators": attenuators,
        "inputs": first["inputs"],
        "outputs": last["outputs"],
    }


def add_ona(app, size: int, *, name: str = "ONA_1"):
    element = add(app, "Optical Network Analyzer", name)
    app.setnamed(element, "number of input ports", int(size))
    app.setnamed(element, "input parameter", "start and stop")
    app.setnamed(element, "start frequency", 193.09e12)
    app.setnamed(element, "stop frequency", 193.11e12)
    app.setnamed(element, "number of points", 3)
    app.setnamed(element, "analysis type", "scattering data")
    for prop, value in (("minimum amplitude", 1e-30), ("minimum loss", 300.0),
                        ("sensitivity", -300.0)):
        try:
            app.setnamed(element, prop, value)
        except Exception:
            pass
    return element


class ONAResultMissing(RuntimeError):
    """The ONA provider omitted an exact-zero transmission result."""


class ONAResultInvalid(RuntimeError):
    """The ONA returned malformed, empty, or non-finite data."""


def _transmission(app, ona: str, index: int) -> complex:
    """Read one ONA transmission, classifying missing vs invalid data.

    Only ``ONAResultMissing`` is eligible for a separately measured power
    fallback.  API errors, missing fields, empty arrays and non-finite values
    raise ``ONAResultInvalid`` and must abort the experiment.
    """
    result_name = f"input {index}/mode 1/transmission"
    try:
        data = app.getresult(ona, result_name)
    except Exception as exc:
        message = str(exc).lower()
        if "can not find result" in message or "not found" in message:
            raise ONAResultMissing(
                f"ONA omitted {result_name} from {ona}") from exc
        raise ONAResultInvalid(
            f"failed to read ONA result {result_name} from {ona}") from exc
    if not isinstance(data, dict):
        raise ONAResultInvalid(f"ONA result {result_name} is not a dataset")
    if "TE transmission" not in data:
        raise ONAResultInvalid(
            f"ONA result {result_name} has no 'TE transmission' field")
    try:
        values = np.asarray(data["TE transmission"], dtype=np.complex128).ravel()
    except (TypeError, ValueError) as exc:
        raise ONAResultInvalid(f"ONA result {result_name} is not a complex array") from exc
    if values.size == 0:
        raise ONAResultInvalid(f"ONA result {result_name} is empty")
    if not np.all(np.isfinite(values)):
        raise ONAResultInvalid(f"ONA result {result_name} contains non-finite values")
    return complex(values[values.size // 2])


class PowerZeroProver:
    """Independent time-domain power proof for an ONA missing result.

    ONA omits an exact-zero transmission.  This helper builds the same DUT
    without an ONA, drives one input at a time with a CW laser, and reads a
    power meter at the requested output.  A missing ONA result is accepted as
    zero only when this independent power is below a strict floor.
    """

    def __init__(self, build, size: int, *, power_limit_w: float = 1e-12):
        self.build = build
        self.size = int(size)
        self.power_limit_w = float(power_limit_w)
        self.app = None
        self.model = None
        self.cw = None
        self.meters = []
        self.current_input = None
        self.last_powers = None

    def _ensure(self):
        if self.app is not None:
            return
        app = new_app()
        app.set("time window", 5e-9)
        app.set("sample rate", 200e9)
        cw = add(app, "CW Laser", "ZP_CW")
        app.setnamed(cw, "power", 1.0)
        app.setnamed(cw, "frequency", 193.1e12)
        app.setnamed(cw, "linewidth", 0.0)
        model = self.build(app)
        if len(model["inputs"]) != self.size or len(model["outputs"]) != self.size:
            app.close()
            raise ValueError("zero-prover DUT size does not match requested size")
        meters = [add(app, "Optical Power Meter", f"ZP_PM{j}") for j in range(self.size)]
        for meter in meters:
            app.setnamed(meter, "power unit", "W")
            app.setnamed(meter, "sensitivity", 1e-13)
            app.setnamed(meter, "limit time range", True)
            app.setnamed(meter, "start time", 3e-9)
            app.setnamed(meter, "stop time", 5e-9)
        for j, output in enumerate(model["outputs"]):
            connect(app, output, Port(meters[j], "input"))
        self.app, self.model, self.cw, self.meters = app, model, cw, meters

    def _run_input(self, j: int):
        self._ensure()
        if self.current_input == j:
            return
        self.app.switchtodesign()
        if self.current_input is not None:
            previous = self.model["inputs"][self.current_input]
            self.app.disconnect(self.cw, "output", previous.element, previous.port)
        connect(self.app, Port(self.cw, "output"), self.model["inputs"][j])
        self.app.run()
        self.current_input = j
        self.last_powers = [
            float(np.asarray(self.app.getresult(meter, "mode 1/power")).ravel()[0])
            for meter in self.meters
        ]
        self.app.switchtodesign()

    def measure_all(self):
        """Return an N×N power matrix before the ONA app is run."""
        self._ensure()
        powers = np.zeros((self.size, self.size), dtype=float)
        for j in range(self.size):
            self._run_input(j)
            powers[:, j] = self.last_powers
        return powers

    def prove_zero(self, j: int, i: int) -> bool:
        self._run_input(j)
        power = self.last_powers[i]
        return bool(math.isfinite(power) and power <= self.power_limit_w)

    def close(self):
        if self.app is not None:
            self.app.close()
            self.app = None


def extract_forward(app, ona: str, inputs, outputs, *, zero_power_map=None,
                    diagnostics=None) -> np.ndarray:
    """Return H[out, in] by exciting each left input and reading all right outputs.

    Only a genuinely omitted exact-zero ONA result may use the independent
    power map.  Malformed data, API failures and non-finite values abort.
    ``diagnostics`` (if provided) receives fallback entries and the power limit;
    fallback entries are *detection-limited zeros*, not measured complex zeros.
    """
    n = len(inputs)
    if len(outputs) != n:
        raise ValueError("testbench requires the same number of inputs and outputs")
    if zero_power_map is not None:
        zero_power_map = np.asarray(zero_power_map, dtype=float)
        if zero_power_map.shape != (n, n):
            raise ValueError("zero power map must have shape (N,N)")
        if not np.all(np.isfinite(zero_power_map)) or np.any(zero_power_map < 0):
            raise ValueError("zero power map must be finite and nonnegative")
    fallbacks = []
    H = np.zeros((n, n), dtype=np.complex128)
    # A previous call (for example when reusing one MZI for several phase
    # settings) leaves the ONA source attached to its last input.  Clear every
    # possible source connection before starting a new sweep.
    app.switchtodesign()
    for candidate in inputs:
        disconnect(app, Port(ona, "output"), candidate)
    previous = None
    for j in range(n):
        if previous is not None:
            app.switchtodesign()
            disconnect(app, Port(ona, "output"), previous)
        connect(app, Port(ona, "output"), inputs[j])
        app.run()
        for i in range(n):
            try:
                H[i, j] = _transmission(app, ona, i + 1)
            except ONAResultMissing as exc:
                if zero_power_map is None:
                    raise RuntimeError(
                        f"ONA output ({i},{j}) missing and no power proof") from exc
                power = float(zero_power_map[i, j])
                limit = 1e-12
                if power > limit:
                    raise RuntimeError(
                        f"ONA output ({i},{j}) missing; power {power:.3e} W "
                        f"exceeds zero limit {limit:.1e} W") from exc
                amplitude_upper = float(np.sqrt(power / 1.0))
                fallbacks.append({
                    "output": int(i),
                    "input": int(j),
                    "power_w": power,
                    "power_limit_w": limit,
                    "amplitude_upper_bound": amplitude_upper,
                    "status": "below_detection_limit",
                })
                H[i, j] = 0j
        previous = inputs[j]
    if not np.all(np.isfinite(H)):
        raise RuntimeError("forward matrix contains non-finite values")
    if diagnostics is not None:
        diagnostics["fallbacks"] = fallbacks
        diagnostics["complex_fully_measured"] = not fallbacks
        diagnostics["power_limit_w"] = 1e-12
        diagnostics["power_meter_sensitivity_w"] = 1e-13
        diagnostics["amplitude_upper_bound"] = (
            max((item["amplitude_upper_bound"] for item in fallbacks), default=0.0))
    return H


def relative_error(actual, reference) -> float:
    actual = np.asarray(actual, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(actual - reference) / denominator) if denominator else float("nan")


def absolute_error(actual, reference) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def save_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def new_app(*, hide: bool = True):
    lumapi = load_lumapi()
    return lumapi.INTERCONNECT(hide=hide)


def configure_frequency_testbench(app, size: int):
    """Create an ONA with N inputs and return (app, ona)."""
    ona = add_ona(app, size)
    return ona


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
