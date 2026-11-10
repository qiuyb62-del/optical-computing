"""Minimal 2D-slab FDTD diagnostic (not a 500x220 nm strip waveguide).

This entry is deliberately separate from the old ``run_device_fdtd.py``.
The only geometry change in the first run is extending Si through both PML
regions; the result remains a z-invariant slab and is never called a strip
waveguide.

A ratio of two source-normalized net-flux monitors is used so the common
normalization cancels: ``T_net_ratio = output_flux / input_flux``.  The values
are dimensionless, not watts.  The script records all mesh levels and applies
a strict pointwise passive check; ``geometry_valid_for_strip`` is always false.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumerical.interconnect import app_version, load_lumapi, save_json, timestamp  # noqa: E402
from lumerical.validation import power_ratio_mesh_converged, strict_passivity  # noqa: E402


def run_case(mesh_accuracy: int, outdir: Path, *, pml_layers: int = 16,
             simulation_time: float = 10e-12, auto_shutoff: float = 1e-9):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    app = load_lumapi().FDTD(hide=True)
    try:
        app.addfdtd()
        for prop, value in (
            ("dimension", "2D"),
            ("x span", 12e-6),
            ("y span", 4e-6),
            ("mesh accuracy", mesh_accuracy),
            ("background material", "SiO2 (Glass) - Palik"),
            ("simulation time", simulation_time),
            ("auto shutoff min", auto_shutoff),
            ("pml layers", pml_layers),
        ):
            app.set(prop, value)
        app.addrect()
        for prop, value in (
            ("name", "Si_slab"), ("x", 0), ("y", 0),
            # 16 um > 12 um domain: the slab passes through both PML regions.
            ("x span", 16e-6), ("y span", 0.5e-6), ("z span", 0),
            ("material", "Si (Silicon) - Palik"),
        ):
            app.set(prop, value)
        app.addmode()
        for prop, value in (
            ("injection axis", "x-axis"), ("direction", "Forward"),
            # Keep the source upstream of the input reference plane; a monitor
            # exactly on the source plane measures a non-positive net flux.
            ("x", -3.5e-6), ("y", 0), ("y span", 2.5e-6),
            ("wavelength start", 1.55e-6), ("wavelength stop", 1.56e-6),
            ("mode selection", "fundamental TE"),
        ):
            app.set(prop, value)
        monitors = []
        for x, name in ((-3e-6, "P_in"), (3e-6, "P_out")):
            app.addpower()
            actual = app.get("name")
            for prop, value in (("monitor type", "2D X-normal"), ("x", x),
                                ("y", 0), ("y span", 2.5e-6)):
                app.set(prop, value)
            monitors.append(actual)
        app.save(str(outdir / f"fdtd_slab_mesh{mesh_accuracy}.fsp"))
        app.run()
        values = []
        wavelength_arrays = []
        for monitor in monitors:
            data = app.getresult(monitor, "T")
            wavelength_arrays.append(np.asarray(data["lambda"], dtype=float).ravel())
            values.append(np.asarray(data["T"], dtype=float).ravel())
        source_neff = complex(np.asarray(
            app.getresult("source", "neff")["neff"]).ravel()[0])
        if values[0].size == 0 or values[1].size != values[0].size:
            raise RuntimeError("power monitors returned empty or mismatched arrays")
        if not np.array_equal(wavelength_arrays[0], wavelength_arrays[1]):
            raise RuntimeError("input/output monitors returned different wavelength grids")
        if not np.all(np.isfinite(values[0])) or not np.all(np.isfinite(values[1])):
            raise RuntimeError("non-finite power monitor data")
        if np.any(values[0] <= 0):
            raise RuntimeError("input monitor flux is non-positive")
        ratio = values[1] / values[0]
        return {
            "wavelength_um": (wavelength_arrays[0] * 1e6).tolist(),
            "input_flux_normalized": values[0].tolist(),
            "output_flux_normalized": values[1].tolist(),
            "T_net_ratio": ratio.tolist(),
            "normalization": "dimensionless source-normalized net flux; not watts",
            "source_neff": {"real": source_neff.real, "imag": source_neff.imag},
            "pml_layers": pml_layers,
            "simulation_time_s": simulation_time,
            "auto_shutoff": auto_shutoff,
        }
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "progress/07_lumerical/device_fdtd_slab")
    parser.add_argument("--meshes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--pml-layers", type=int, default=16)
    parser.add_argument("--simulation-time", type=float, default=10e-12)
    parser.add_argument("--auto-shutoff", type=float, default=1e-9)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for mesh in args.meshes:
        print(f"running slab mesh accuracy {mesh} ...", flush=True)
        runs[str(mesh)] = run_case(mesh, args.out,
                                   pml_layers=args.pml_layers,
                                   simulation_time=args.simulation_time,
                                   auto_shutoff=args.auto_shutoff)
    meshes = sorted(runs, key=int)
    all_t = [np.asarray(runs[m]["T_net_ratio"], dtype=float) for m in meshes]
    all_values = np.concatenate([t.ravel() for t in all_t])
    max_t_all = float(np.max(all_values))
    passive = strict_passivity(all_values, upper=1.0)
    mesh_check = power_ratio_mesh_converged(runs, meshes, threshold=1e-3,
                                            minimum_meshes=3)
    power_mesh_checks_passed = bool(passive["passed"] and mesh_check["passed"])
    payload = {
        "stage": "device_fdtd_slab_diagnostic",
        "status": "2D z-invariant slab diagnostic; not a strip-waveguide validation",
        "criteria_version": "strict_pointwise_T_le_1_and_three_mesh_power_ratio_v1",
        "code_commit": "working-tree",
        "timestamp": timestamp(),
        "lumerical": app_version(),
        "simulated_geometry": {
            "dimension": "2D (z invariant)",
            "si_width_y_um": 0.5,
            "si_thickness_z": "infinite (z invariant)",
            "si_length_x_um": 16.0,
            "domain_x_um": 12.0,
            "note": "z span is deliberately 0; this cannot represent 220 nm thickness",
        },
        "intended_target": "0.5 x 0.22 um SOI strip waveguide",
        "geometry_valid_for_strip": False,
        "diagnostic_only": True,
        "monitor_definition": "dimensionless T_net_ratio = output_flux_normalized / input_flux_normalized at x=+/-3 um",
        "meshes": [int(m) for m in meshes],
        "runs": runs,
        "max_T_all": max_t_all,
        "strict_passivity": passive,
        "raw_passive": passive["passed"],
        "mesh_pairwise_delta": mesh_check["pairwise"],
        "power_ratio_mesh_converged": mesh_check["passed"],
        "mesh_converged": mesh_check["passed"],
        "phase_converged": None,
        "pml_domain_converged": None,
        "time_converged": None,
        "mode_identity_valid": None,
        "power_mesh_checks_passed": power_mesh_checks_passed,
        "complete": False,
        "slab_diagnostic_complete": False,
        "slab_diagnostic_complete_reason": "PML, time, mode identity and complex S21 checks are not all measured",
        "limitation": "A converged slab diagnostic still does not validate a 500x220 nm strip waveguide.",
    }
    save_json(args.out / "device_fdtd_slab.json", payload)
    print(json.dumps({k: payload[k] for k in
                      ("max_T_all", "raw_passive", "power_ratio_mesh_converged",
                       "power_mesh_checks_passed", "slab_diagnostic_complete",
                       "geometry_valid_for_strip")},
                     indent=2), flush=True)
    return 0 if payload["raw_passive"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
