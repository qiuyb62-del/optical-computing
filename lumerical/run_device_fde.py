"""Device-level SOI waveguide simulation with MODE FDE and circuit S-parameter check.

Geometry: 500 nm x 220 nm Si strip on SiO2, 2D X-normal FDE cross-section,
fundamental TE mode, 1530-1570 nm.  The FDE effective index is converted to the
complex propagation S21 of a 100 um waveguide and compared with an INTERCONNECT
Straight Waveguide configured from the same index.  This is a representative
device check; it is not a 3D FDTD of the 240-MZI network.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumerical.interconnect import (  # noqa: E402
    Port, add, add_ona, app_version, connect, new_app, save_json, timestamp,
)
from lumerical.interconnect import load_lumapi  # noqa: E402
from lumerical.validation import fde_phase_metrics  # noqa: E402

C0 = 299792458.0


def build_fde(mode_app):
    mode_app.addfde()
    for prop, value in (
        ("solver type", "2D X normal"),
        ("y span", 3e-6),
        ("z span", 2e-6),
        ("mesh cells y", 80),
        ("mesh cells z", 80),
        ("background material", "SiO2 (Glass) - Palik"),
    ):
        mode_app.set(prop, value)
    mode_app.addrect()
    for prop, value in (
        ("name", "Si_WG"), ("x", 0), ("y", 0), ("z", 0),
        ("y span", 0.5e-6), ("z span", 0.22e-6),
        ("material", "Si (Silicon) - Palik"),
    ):
        mode_app.set(prop, value)


def solve_one(mode_app, wavelength, cells):
    mode_app.switchtolayout()
    mode_app.select("FDE")
    mode_app.set("mesh cells y", int(cells))
    mode_app.set("mesh cells z", int(cells))
    mode_app.set("wavelength", float(wavelength))
    found = mode_app.findmodes()
    if found < 1:
        raise RuntimeError("FDE found no mode")
    neff = complex(np.asarray(mode_app.getresult("FDE::data::mode1", "neff"))[0, 0])
    te = float(np.asarray(mode_app.getresult("FDE::data::mode1", "TE polarization fraction")).ravel()[0])
    return neff, te


def circuit_s21(neff, length, wavelength, frequency):
    mode_app = None
    app = new_app()
    try:
        ona = add_ona(app, 1, name="ONA_1")
        app.setnamed(ona, "input parameter", "start and stop")
        app.setnamed(ona, "start frequency", frequency - 20e9)
        app.setnamed(ona, "stop frequency", frequency + 20e9)
        app.setnamed(ona, "number of points", 3)
        wg = add(app, "Straight Waveguide", "FDE_WG")
        app.setnamed(wg, "length", float(length))
        app.setnamed(wg, "frequency", float(frequency))
        app.setnamed(wg, "effective index 1", float(neff.real))
        app.setnamed(wg, "effective index 2", float(neff.real))
        app.setnamed(wg, "group index 1", float(neff.real))
        app.setnamed(wg, "group index 2", float(neff.real))
        app.setnamed(wg, "loss 1", 0.0)
        connect(app, Port(ona, "output"), Port(wg, "port 1"))
        connect(app, Port(wg, "port 2"), Port(ona, "input 1"))
        app.run()
        data = app.getresult(ona, "input 1/mode 1/transmission")
        values = np.asarray(data["TE transmission"], dtype=complex).ravel()
        frequencies = np.asarray(data["frequency"], dtype=float).ravel()
        # ONA start/stop spacing is (stop-start)/N, so the middle sample is not
        # exactly the requested center.  Return it instead of silently assuming.
        return complex(values[len(values) // 2]), float(frequencies[len(frequencies) // 2])
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "progress/07_lumerical/device_fde")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    wavelengths = np.array([1.53, 1.54, 1.55, 1.56, 1.57]) * 1e-6
    lumapi = load_lumapi()
    mode_app = lumapi.MODE(hide=True)
    try:
        build_fde(mode_app)
        # Save a compact, geometry-only native project before solving.  Saving
        # after a 480x480 solve embeds hundreds of MB of mode data.
        mode_app.save(str(args.out / "soi_waveguide_fde.lms"))
        coarse = [solve_one(mode_app, lam, 60) for lam in wavelengths]
        fine = [solve_one(mode_app, lam, 120) for lam in wavelengths]
        finer = [solve_one(mode_app, lam, 480) for lam in wavelengths]
        mode_app.switchtolayout()
        mode_app.select("FDE")
        mode_app.set("wavelength", 1.55e-6)
        mode_app.set("mesh cells y", 480)
        mode_app.set("mesh cells z", 480)
        mode_app.findmodes()
        neff_1550 = complex(np.asarray(mode_app.getresult("FDE::data::mode1", "neff"))[0, 0])
    finally:
        mode_app.close()

    length = 100e-6
    frequency = C0 / 1.55e-6
    s21_circuit, actual_frequency = circuit_s21(neff_1550, length, 1.55e-6, frequency)
    actual_wavelength = C0 / actual_frequency
    s21_plus = complex(np.exp(1j * 2 * np.pi * neff_1550.real * length / actual_wavelength))
    s21_minus = complex(np.exp(-1j * 2 * np.pi * neff_1550.real * length / actual_wavelength))
    err_plus = abs(s21_circuit - s21_plus)
    err_minus = abs(s21_circuit - s21_minus)
    use_plus = err_plus <= err_minus
    neff_delta = float(max(abs(a[0] - b[0]) for a, b in zip(fine, finer)))
    phase = fde_phase_metrics(fine[2][0], finer[2][0], length=length,
                              wavelength=1.55e-6, phase_limit=0.01)
    payload = {
        "stage": "device_fde_waveguide",
        "timestamp": timestamp(),
        "lumerical": app_version(),
        "geometry": {"material": "Si (Silicon) - Palik on SiO2 (Glass) - Palik",
                     "width_um": 0.5, "height_um": 0.22, "length_um": 100.0},
        "solver": "MODE FDE, 2D X normal, fundamental TE, y/z span 3/2 um",
        "wavelength_um": (wavelengths * 1e6).tolist(),
        "neff_coarse_60": [{"real": z.real, "imag": z.imag, "te_fraction": te} for z, te in coarse],
        "neff_fine_120": [{"real": z.real, "imag": z.imag, "te_fraction": te} for z, te in fine],
        "neff_finer_480": [{"real": z.real, "imag": z.imag, "te_fraction": te} for z, te in finer],
        "neff_1550": {"real": neff_1550.real, "imag": neff_1550.imag},
        "max_abs_neff_change_60_to_120": float(max(abs(a[0] - b[0]) for a, b in zip(coarse, fine))),
        "max_abs_neff_change_120_to_480": neff_delta,
        "neff_convergence_limit": 3e-3,
        "neff_converged": bool(neff_delta <= 3e-3),
        "phase_convergence": phase,
        "phase_converged": phase["phase_converged"],
        "stage_status": ("complete" if phase["phase_converged"]
                         else "preliminary_phase_not_converged"),
        "complete": bool(phase["phase_converged"]),
        "s21_from_fde": {"real": s21_plus.real, "imag": s21_plus.imag},
        "s21_interconnect": {"real": s21_circuit.real, "imag": s21_circuit.imag},
        "interconnect_frequency_hz": actual_frequency,
        "interconnect_wavelength_um": actual_wavelength * 1e6,
        "s21_abs_error_plus_convention": float(err_plus),
        "s21_abs_error_minus_convention": float(err_minus),
        "selected_phase_convention": "exp(+i 2 pi neff L / lambda)" if use_plus else "exp(-i 2 pi neff L / lambda)",
        "selected_s21_abs_error": float(min(err_plus, err_minus)),
        "circuit_note": "INTERCONNECT Straight Waveguide configured with the same FDE neff; this checks parameter mapping, not an independent device solution.",
        "fdtd_note": "See run_device_fdtd.py for the separate diagnostic; it is not used for a quantitative S-parameter.",
    }
    save_json(args.out / "device_fde.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

