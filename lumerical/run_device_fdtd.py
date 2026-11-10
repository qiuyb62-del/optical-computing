"""Diagnostic 2D FDTD run for a Si strip waveguide.

This is deliberately a *diagnostic* result, not the source of the circuit
S-parameters.  The earlier apparent 10-minute hangs were traced to post-
processing the large ``farfield`` result after a 4 s solve.  This script reads
only ``T`` and the mode-source ``neff``.

The 10 um Si strip ends inside the domain, so facet reflection produces a
wavelength ripple in T; the coarse/fine difference is reported instead of being
hidden.  The quantitative device S-parameter used in the report comes from the
separate MODE FDE + INTERCONNECT Straight Waveguide check.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumerical.interconnect import app_version, save_json, timestamp  # noqa: E402


def run_case(mesh_accuracy: int, outdir: Path):
    from lumerical.interconnect import load_lumapi
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    app = load_lumapi().FDTD(hide=True)
    try:
        app.addfdtd()
        for prop, value in (
            ("dimension", "2D"), ("x span", 12e-6), ("y span", 4e-6),
            ("mesh accuracy", mesh_accuracy),
            ("background material", "SiO2 (Glass) - Palik"),
            ("simulation time", 5e-12), ("auto shutoff min", 1e-6),
        ):
            app.set(prop, value)
        app.addrect()
        for prop, value in (
            ("name", "Si_WG"), ("x", 0), ("y", 0),
            ("x span", 10e-6), ("y span", 0.5e-6), ("z span", 0),
            ("material", "Si (Silicon) - Palik"),
        ):
            app.set(prop, value)
        app.addmode()
        for prop, value in (
            ("injection axis", "x-axis"), ("direction", "Forward"),
            ("x", -4.8e-6), ("y", 0), ("y span", 2.5e-6),
            ("wavelength start", 1.55e-6), ("wavelength stop", 1.56e-6),
            ("mode selection", "fundamental TE"),
        ):
            app.set(prop, value)
        app.addpower()
        power = app.get("name")
        for prop, value in (("monitor type", "2D X-normal"), ("x", 4.8e-6),
                            ("y", 0), ("y span", 2.5e-6)):
            app.set(prop, value)
        app.save(str(outdir / f"fdtd_diag_mesh{mesh_accuracy}.fsp"))
        app.run()
        # Read only the requested result.  Never enumerate all results here:
        # the farfield dataset is huge and made the process look hung.
        t_data = app.getresult(power, "T")
        source_neff = complex(np.asarray(
            app.getresult("source", "neff")["neff"]).ravel()[0])
        return {
            "wavelength_um": (np.asarray(t_data["lambda"]).ravel() * 1e6).tolist(),
            "T": np.asarray(t_data["T"]).ravel().tolist(),
            "source_neff": {"real": source_neff.real, "imag": source_neff.imag},
        }
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "progress/07_lumerical/device_fdtd")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    coarse = run_case(2, args.out)
    fine = run_case(3, args.out)
    T2 = np.asarray(coarse["T"])
    T3 = np.asarray(fine["T"])
    max_T_all = float(max(np.max(T2), np.max(T3)))
    max_T_fine = float(np.max(T3))
    max_abs_delta = float(np.max(np.abs(T3 - T2)))
    payload = {
        "stage": "device_fdtd_diagnostic",
        "status": "diagnostic only; not used for quantitative circuit S-parameters",
        "diagnostic_only": True,
        "geometry_valid_for_strip": False,
        "simulated_geometry": {
            "dimension": "2D (z invariant)",
            "si_width_y_um": 0.5,
            "si_thickness_z": "infinite (z invariant)",
            "si_length_x_um": 10.0,
            "domain_x_um": 12.0,
        },
        "intended_target": "0.5 x 0.22 um SOI strip waveguide",
        "timestamp": timestamp(),
        "lumerical": app_version(),
        "geometry": {"width_um": 0.5, "height_um": 0.22, "length_um": 10.0,
                     "material": "Si (Silicon) - Palik on SiO2 (Glass) - Palik"},
        "solver": "2D FDTD, fundamental TE mode source, 1550-1560 nm, 5 ps, autoshutoff 1e-6",
        "diagnosis": "Solver completes in about 4 s. Earlier apparent 10-minute hangs came from enumerating all results and retrieving farfield; this script reads only T and source neff.",
        "mesh_accuracy_coarse": 2,
        "mesh_accuracy_fine": 3,
        "coarse": coarse,
        "fine": fine,
        "max_abs_T_change_mesh2_to_mesh3": max_abs_delta,
        "max_T_all": max_T_all,
        "max_T_fine": max_T_fine,
        "max_T": max_T_all,
        "raw_passive": bool(max_T_all <= 1.0),
        "mesh_converged": False,
        "phase_converged": None,
        "pml_domain_converged": None,
        "time_converged": None,
        "complete": False,
        "limitation": "The 10 um strip ends inside the domain; facet reflection and the short run produce a T ripple, so this FDTD diagnostic is not used as a converged device S-parameter.",
    }
    save_json(args.out / "device_fdtd.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
