"""Reproducible Lumerical experiments for the MZI, Clements and SVD stages."""
from __future__ import annotations

import json
from pathlib import Path
import platform

import numpy as np

from .interconnect import (
    Port,
    PowerZeroProver,
    absolute_error,
    add,
    add_mesh,
    add_mzi,
    add_ona,
    add_svd,
    add_svd_nonideal,
    app_version,
    connect,
    extract_forward,
    new_app,
    relative_error,
    save_json,
    timestamp,
)
from .validation import compare_complex

TOL = 1e-6


def _measurement_metadata(diagnostics):
    fallbacks = list(diagnostics.get("fallbacks", []))
    amplitude_upper = float(diagnostics.get(
        "amplitude_upper_bound",
        max((item["amplitude_upper_bound"] for item in fallbacks), default=0.0)))
    return {
        "fallback_count": len(fallbacks),
        "fallback_entries": fallbacks,
        "complex_fully_measured": bool(diagnostics.get("complex_fully_measured", True)),
        "zero_amplitude_upper_bound": amplitude_upper,
        "power_limit_w": float(diagnostics.get("power_limit_w", 1e-12)),
        "power_meter_sensitivity_w": float(
            diagnostics.get("power_meter_sensitivity_w", 1e-13)),
    }


def _zero_power_map(build, size):
    """Independent power map for small circuits; None for the large SVD.

    The 16x16 MNIST W1 is dense, so an ONA zero is not expected.  Building a
    16-channel time-domain prover for every run would be disproportionate.
    """
    if size > 4:
        return None
    prover = PowerZeroProver(build, size)
    try:
        return prover.measure_all()
    finally:
        prover.close()


def _complex_payload(matrix):
    matrix = np.asarray(matrix, dtype=np.complex128)
    return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}


def _case_result(name, H, target, *, extra=None):
    """Build a matrix case without mixing optical and restored scales."""
    H = np.asarray(H, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    if not np.all(np.isfinite(target)):
        raise ValueError("target matrix contains non-finite values")
    check = compare_complex(H, target)
    payload = {
        "name": name,
        "shape": list(H.shape),
        "relative_error": check["relative_error"],
        "absolute_error": check["absolute_error"],
        "criterion": check["criterion"],
        "limit": check["limit"],
        "scale": check.get("scale"),
        "passed": check["passed"],
        "H": _complex_payload(H),
        "target": _complex_payload(target),
    }
    if extra:
        payload.update(extra)
    return payload


def measure_mzi(outdir: Path, *, seed: int = 20261110):
    """Run the standalone 2x2 MZI for theta=0, pi/4, pi/2 and arbitrary phases."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    theta_phi = [(0.0, 0.0), (np.pi / 4, 1.1), (np.pi / 2, -0.8), (0.37, 2.4)]
    app = new_app()
    results = []
    try:
        gate = add_mzi(app, "MZI", 0, 0.0, 0.0)
        ona = add_ona(app, 2)
        for i, output in enumerate(gate["outputs"]):
            connect(app, output, Port(ona, f"input {i + 1}"))
        for theta, phi in theta_phi:
            app.switchtodesign()
            components = gate["components"]
            app.setnamed(components["input_phase"], "phase shift", float(phi))
            app.setnamed(components["arm_phase"], "phase shift", 2.0 * float(theta))
            app.setnamed(components["output_phase_upper"], "phase shift", -float(theta))
            app.setnamed(components["output_phase_lower"], "phase shift", -float(theta))
            power_map = _zero_power_map(
                lambda app, t=theta, p=phi: add_mzi(app, "ZP_MZI", 0, t, p), 2)
            diagnostics = {}
            H = extract_forward(app, ona, gate["inputs"], gate["outputs"],
                                zero_power_map=power_map, diagnostics=diagnostics)
            c, s = np.cos(theta), np.sin(theta)
            target = np.array([[np.exp(1j * phi) * c, -s],
                               [np.exp(1j * phi) * s, c]], dtype=complex)
            case = _case_result(f"theta={theta:.12g},phi={phi:.12g}", H, target,
                                extra={"theta": float(theta), "phi": float(phi),
                                       **_measurement_metadata(diagnostics)})
            case["complete"] = bool(case["passed"] and case["complex_fully_measured"])
            results.append(case)
        app.switchtodesign()
        app.save(str(outdir / "single_mzi.icp"))
        payload = {
            "stage": "lumerical_single_mzi",
            "timestamp": timestamp(),
            "lumerical": app_version(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "seed": seed,
            "convention": "T(theta,phi)=[[exp(i phi)cos theta,-sin theta],[exp(i phi)sin theta,cos theta]]",
            "coupler_mapping": "input coupler conjugate=0 gives B; output coupler conjugate=1 gives B†",
            "phase_mapping": "input phi; upper arm delta=2theta; output chi_upper=chi_lower=-theta",
            "cases": results,
            "passed": all(case["passed"] for case in results),
            "complete": all(case["complete"] for case in results),
        }
        save_json(outdir / "single_mzi.json", payload)
        print(f"MZI: {sum(c['passed'] for c in results)}/{len(results)} cases pass; "
              f"complete={payload['complete']}; "
              f"max error={max(c['relative_error'] or 0 for c in results):.3e}")
        return payload
    finally:
        app.close()


def measure_unitary(U, outdir: Path, name: str):
    """Build a 4-channel Clements mesh from the project decomposition and measure H."""
    from src.clements import decompose_clements

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ops, phases = decompose_clements(np.asarray(U, dtype=complex))
    app = new_app()
    try:
        mesh = add_mesh(app, "CLEMENTS", ops, phases)
        ona = add_ona(app, len(phases))
        for i, output in enumerate(mesh["outputs"]):
            connect(app, output, Port(ona, f"input {i + 1}"))
        power_map = _zero_power_map(
            lambda app: add_mesh(app, "ZP_MESH", ops, phases), len(phases))
        diagnostics = {}
        H = extract_forward(app, ona, mesh["inputs"], mesh["outputs"],
                            zero_power_map=power_map, diagnostics=diagnostics)
        app.switchtodesign()
        app.save(str(outdir / f"{name}.icp"))
        case = _case_result(name, H, U, extra={
            "size": int(len(phases)),
            "mzi_count": len(ops),
            "output_phases": [float(x) for x in phases],
            **_measurement_metadata(diagnostics),
        })
        case["complete"] = bool(case["passed"] and case["complex_fully_measured"])
        return case
    finally:
        app.close()


def measure_zero_svd(W, outdir: Path, name: str):
    """Prove the zero matrix with independent time-domain power meters.

    ONA omits an exact-zero transmission result even at minimum amplitude 0, so
    a missing ONA value cannot be used as zero.  Here a CW source drives each
    input in turn and four power meters verify that every output is below a
    strict power floor.  This is a different testbench from the complex ONA
    extraction, and the JSON says so.
    """
    from src.svd_matmul import compile_svd

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    W = np.asarray(W, dtype=complex)
    if np.linalg.norm(W) != 0:
        raise ValueError("measure_zero_svd requires an all-zero matrix")
    circuit = compile_svd(W)
    app = new_app()
    try:
        app.set("time window", 5e-9)
        app.set("sample rate", 200e9)
        cw = add(app, "CW Laser", "ZERO_CW")
        app.setnamed(cw, "power", 1.0)
        app.setnamed(cw, "frequency", 193.1e12)
        app.setnamed(cw, "linewidth", 0.0)
        svd = add_svd(app, "ZERO_SVD", circuit)
        meters = [add(app, "Optical Power Meter", f"ZERO_PM{j}")
                  for j in range(circuit.size)]
        for meter in meters:
            app.setnamed(meter, "power unit", "W")
            app.setnamed(meter, "sensitivity", 1e-13)
            app.setnamed(meter, "limit time range", True)
            app.setnamed(meter, "start time", 3e-9)
            app.setnamed(meter, "stop time", 5e-9)
        for j, output in enumerate(svd["outputs"]):
            connect(app, output, Port(meters[j], "input"))
        powers = []
        for j in range(circuit.size):
            if j:
                app.switchtodesign()
                app.disconnect(cw, "output", svd["inputs"][j - 1].element,
                               svd["inputs"][j - 1].port)
            connect(app, Port(cw, "output"), svd["inputs"][j])
            app.run()
            row = []
            for meter in meters:
                value = app.getresult(meter, "mode 1/power")
                row.append(float(np.asarray(value).ravel()[0]))
            if not np.all(np.isfinite(row)):
                raise RuntimeError("zero-power monitor returned non-finite data")
            powers.append(row)
        app.switchtodesign()
        app.save(str(outdir / f"{name}.icp"))
        max_power = float(np.max(np.abs(np.asarray(powers))))
        zero_limit = 1e-12
        input_power = 1.0
        amplitude_upper = float(np.sqrt(max_power / input_power))
        zero_passed = bool(max_power <= zero_limit)
        return {
            "name": name,
            "shape": [int(circuit.size), int(circuit.size)],
            "criterion": "power_leakage",
            "measurement": "time-domain power meters (independent zero proof)",
            "input_power_w": input_power,
            "zero_power_w": powers,
            "max_zero_power_w": max_power,
            "zero_power_limit_w": zero_limit,
            "power_meter_sensitivity_w": 1e-13,
            "amplitude_upper_bound": amplitude_upper,
            "zero_passed": zero_passed,
            "passed": zero_passed,
            "complex_H_measured": False,
            "complex_fully_measured": False,
            "relative_error": None,
            "absolute_error": None,
            "restored_relative_error": None,
            "restored_absolute_error": None,
            "restored_passed": None,
            "complete": False,
        }
    finally:
        app.close()


def measure_svd(W, outdir: Path, name: str):
    """Build V† -> attenuators -> U and measure the optical transfer H≈W/s."""
    from src.svd_matmul import compile_svd

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    W = np.asarray(W, dtype=complex)
    if np.linalg.norm(W) == 0:
        return measure_zero_svd(W, outdir, name)
    circuit = compile_svd(W)
    app = new_app()
    try:
        svd = add_svd(app, "SVD", circuit)
        ona = add_ona(app, circuit.size)
        for i, output in enumerate(svd["outputs"]):
            connect(app, output, Port(ona, f"input {i + 1}"))
        power_map = _zero_power_map(
            lambda app: add_svd(app, "ZP_SVD", circuit), circuit.size)
        diagnostics = {}
        H = extract_forward(app, ona, svd["inputs"], svd["outputs"],
                            zero_power_map=power_map, diagnostics=diagnostics)
        app.switchtodesign()
        app.save(str(outdir / f"{name}.icp"))
        restored = circuit.recovery_scale * H
        restored_check = compare_complex(restored, W)
        case = _case_result(name, H, W / circuit.recovery_scale, extra={
            "size": int(circuit.size),
            "mzi_count": 2 * circuit.size * (circuit.size - 1) // 2,
            "recovery_scale": float(circuit.recovery_scale),
            "singular_values": [float(x) for x in circuit.singular_values],
            "amplitude_transmissions": [float(x) for x in circuit.amplitude_transmissions],
            "restored_H": _complex_payload(restored),
            "restored_relative_error": restored_check["relative_error"],
            "restored_absolute_error": restored_check["absolute_error"],
            "restored_criterion": restored_check["criterion"],
            "restored_passed": restored_check["passed"],
            **_measurement_metadata(diagnostics),
        })
        case["complete"] = bool(
            case["passed"] and case.get("restored_passed", True)
            and case["complex_fully_measured"])
        return case
    finally:
        app.close()


def measure_svd_nonideal(W, outdir: Path, name: str, config, *, seed: int):
    """Build a fixed nonideal device from Python's snapshot and measure H."""
    from src.nonideal import realize_nonideal, reconstruct_nonideal
    from src.svd_matmul import compile_svd

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    circuit = compile_svd(np.asarray(W, dtype=complex))
    device = realize_nonideal(circuit, config, seed=seed)
    H_python_optical = reconstruct_nonideal(device, restore_scale=False)
    H_python_restored = reconstruct_nonideal(device, restore_scale=True)
    app = new_app()
    try:
        svd = add_svd_nonideal(app, "SVD_NI", device)
        ona = add_ona(app, device.size)
        for i, output in enumerate(svd["outputs"]):
            connect(app, output, Port(ona, f"input {i + 1}"))
        power_map = _zero_power_map(
            lambda app: add_svd_nonideal(app, "ZP_NI", device), device.size)
        diagnostics = {}
        H = extract_forward(app, ona, svd["inputs"], svd["outputs"],
                            zero_power_map=power_map, diagnostics=diagnostics)
        app.switchtodesign()
        app.save(str(outdir / f"{name}.icp"))
        restored = device.recovery_scale * H
        restored_check = compare_complex(restored, H_python_restored)
        case = _case_result(
            name, H, H_python_optical,
            extra={
                "size": int(device.size),
                "seed": int(seed),
                "config": {
                    "phase_std_rad": float(config.phase_std_rad),
                    "loss_db_per_mzi": float(config.loss_db_per_mzi),
                    "phase_bits": config.phase_bits,
                    "coupler_through": float(config.coupler_through),
                    "upper_arm_loss_db": float(config.upper_arm_loss_db),
                },
                "recovery_scale": float(device.recovery_scale),
                "restored_H": _complex_payload(restored),
                "restored_relative_error": restored_check["relative_error"],
                "restored_absolute_error": restored_check["absolute_error"],
                "restored_criterion": restored_check["criterion"],
                "restored_passed": restored_check["passed"],
                **_measurement_metadata(diagnostics),
            })
        case["complete"] = bool(
            case["passed"] and case.get("restored_passed", True)
            and case["complex_fully_measured"])
        return case
    finally:
        app.close()


def write_clements_report(outdir: Path, cases, *, seed: int):
    payload = {
        "stage": "lumerical_clements4",
        "timestamp": timestamp(),
        "lumerical": app_version(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": seed,
        "tol": TOL,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
        "complete": all(case.get("complete", case["passed"]) for case in cases),
    }
    save_json(Path(outdir) / "clements4.json", payload)
    return payload


def write_svd_report(outdir: Path, cases, *, seed: int):
    payload = {
        "stage": "lumerical_svd4",
        "timestamp": timestamp(),
        "lumerical": app_version(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": seed,
        "tol": TOL,
        "cases": cases,
        "passed": all(case["passed"] and case.get("restored_passed", True) is not False
                      for case in cases),
        "complete": all(case.get("complete", False) for case in cases),
    }
    save_json(Path(outdir) / "svd4.json", payload)
    return payload
