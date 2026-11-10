"""Time-domain coherent testbench for the 4x4 SVD circuit.

One CW laser is coherently fanned out (INTERCONNECT's Fork 1xN is an ideal
field copy, not a physical 1/N power splitter) to four amplitude/phase encoded
inputs and to eight local-oscillator paths.  Each SVD output drives two 2x2
couplers: one with a 0 rad LO and one with a pi/2 LO.  Four photocurrents per
output therefore give I and Q after balanced subtraction.  This is a receiver
testbench; the DUT itself remains the passive SVD circuit built from local
couplers, phase shifters and attenuators.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .interconnect import (
    Port,
    add,
    add_ona,
    add_svd,
    connect,
    coupler,
    new_app,
)


def _set_if_possible(app, name, prop, value):
    try:
        app.setnamed(name, prop, value)
    except Exception:
        pass


def build_td_svd(app, circuit, prefix: str = "TD"):
    """Build source encoding, SVD DUT and coherent I/Q receiver."""
    app.set("time window", 5e-9)
    app.set("sample rate", 200e9)

    cw = add(app, "CW Laser", f"{prefix}_CW")
    for prop, value in (("power", 1.0), ("frequency", 193.1e12), ("linewidth", 0.0), ("phase", 0.0)):
        app.setnamed(cw, prop, value)

    froot = add(app, "Fork 1xN", f"{prefix}_FANROOT")
    app.setnamed(froot, "number of ports", 2)
    fin = add(app, "Fork 1xN", f"{prefix}_FANIN")
    app.setnamed(fin, "number of ports", int(circuit.size))
    flo = add(app, "Fork 1xN", f"{prefix}_FANLO")
    app.setnamed(flo, "number of ports", int(2 * circuit.size))
    connect(app, Port(cw, "output"), Port(froot, "input"))
    connect(app, Port(froot, "output 1"), Port(fin, "input"))
    connect(app, Port(froot, "output 2"), Port(flo, "input"))

    svd = add_svd(app, f"{prefix}_DUT", circuit)
    controls = []
    for j in range(circuit.size):
        att = add(app, "Optical Attenuator", f"{prefix}_XATT{j}")
        ph = add(app, "Optical Phase Shift", f"{prefix}_XPH{j}")
        connect(app, Port(fin, f"output {j + 1}"), Port(att, "port 1"))
        connect(app, Port(att, "port 2"), Port(ph, "port 1"))
        connect(app, Port(ph, "port 2"), svd["inputs"][j])
        controls.append((att, ph))

    receiver = []
    for j in range(circuit.size):
        fout = add(app, "Fork 1xN", f"{prefix}_FOUT{j}")
        app.setnamed(fout, "number of ports", 2)
        connect(app, svd["outputs"][j], Port(fout, "input"))
        h0 = coupler(app, f"{prefix}_H0_{j}", conjugate=0)
        h90 = coupler(app, f"{prefix}_H90_{j}", conjugate=0)
        connect(app, Port(fout, "output 1"), Port(h0, "port 1"))
        connect(app, Port(fout, "output 2"), Port(h90, "port 1"))
        connect(app, Port(flo, f"output {2 * j + 1}"), Port(h0, "port 2"))
        lo90 = add(app, "Optical Phase Shift", f"{prefix}_LO90_{j}")
        app.setnamed(lo90, "phase shift", float(np.pi / 2))
        connect(app, Port(flo, f"output {2 * j + 2}"), Port(lo90, "port 1"))
        connect(app, Port(lo90, "port 2"), Port(h90, "port 2"))
        pds, oscs = [], []
        for branch, hybrid in (("I", h0), ("Q", h90)):
            for k, port in enumerate(("port 3", "port 4")):
                pd = add(app, "Pin Photodetector", f"{prefix}_PD_{j}_{branch}{k}")
                _set_if_possible(app, pd, "responsivity", 1.0)
                _set_if_possible(app, pd, "dark current", 0.0)
                _set_if_possible(app, pd, "enable shot noise", False)
                _set_if_possible(app, pd, "enable thermal noise", False)
                osc = add(app, "Oscilloscope", f"{prefix}_OSC_{j}_{branch}{k}")
                connect(app, Port(hybrid, port), Port(pd, "input"))
                connect(app, Port(pd, "output"), Port(osc, "input"))
                pds.append(pd)
                oscs.append(osc)
        receiver.append({"output": j, "pds": pds, "oscilloscopes": oscs})
    return {
        "cw": cw,
        "fanin": fin,
        "fanlo": flo,
        "svd": svd,
        "controls": controls,
        "receiver": receiver,
    }


def set_encoded_vector(app, controls, x):
    """Set x/g on the four amplitude/phase encoders and return g."""
    x = np.asarray(x, dtype=complex)
    if x.shape != (len(controls),):
        raise ValueError("input vector length must match the circuit")
    g = max(1.0, float(np.max(np.abs(x))) if len(x) else 1.0)
    scaled = x / g
    for (att, ph), value in zip(controls, scaled):
        amplitude = float(abs(value))
        db = 300.0 if amplitude == 0.0 else -20.0 * math.log10(amplitude)
        app.setnamed(att, "attenuation", db)
        app.setnamed(ph, "phase shift", float(np.angle(value)) if amplitude else 0.0)
    return g


def _mean_signal(app, osc: str, settle: int = 200) -> float:
    data = app.getresult(osc, "signal")
    values = np.asarray(data["amplitude (a.u.)"], dtype=float).ravel()
    if values.size == 0:
        return 0.0
    return float(values[-min(settle, values.size):].mean())


def read_iq(app, receiver) -> np.ndarray:
    """Return the complex field after the 0/90 degree hybrid and balanced math."""
    output = []
    for item in receiver:
        values = [_mean_signal(app, osc) for osc in item["oscilloscopes"]]
        # values: [I+, I-, Q+, Q-]
        d_i = values[0] - values[1]
        d_q = values[2] - values[3]
        # With unit LO amplitude: I = -d_q/2, Q = d_i/2.
        output.append((-d_q / 2.0) + 1j * (d_i / 2.0))
    return np.asarray(output, dtype=complex)


def run_vector(app, testbench, x):
    app.switchtodesign()
    g = set_encoded_vector(app, testbench["controls"], x)
    app.run()
    return read_iq(app, testbench["receiver"]), g


def measure_svd_time_domain(W, outdir: Path, vectors, *, name: str = "svd4_td"):
    """Build a 4x4 SVD, run a sequence of encoded vectors, and save raw complex outputs."""
    from src.svd_matmul import compile_svd

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    circuit = compile_svd(np.asarray(W, dtype=complex))
    app = new_app()
    records = []
    try:
        testbench = build_td_svd(app, circuit, "TD")
        for index, x in enumerate(vectors):
            y_measured, g = run_vector(app, testbench, x)
            x = np.asarray(x, dtype=complex)
            records.append({
                "index": index,
                "x": {"real": x.real.tolist(), "imag": x.imag.tolist()},
                "encoding_scale_g": float(g),
                "measured_optical_output": {
                    "real": y_measured.real.tolist(), "imag": y_measured.imag.tolist()},
            })
            print(f"vector {index}: g={g:.6g} |y|={np.abs(y_measured)}", flush=True)
        app.switchtodesign()
        app.save(str(outdir / f"{name}.icp"))
        return {"circuit": circuit, "records": records, "testbench": testbench}
    finally:
        app.close()
