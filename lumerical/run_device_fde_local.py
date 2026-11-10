"""Local-mesh FDE convergence check with explicit mode tracking.

This is a full-vector 2D X-normal FDE of an x-uniform 500 x 220 nm Si strip in
SiO2.  It is not the z-invariant 2D FDTD slab.

The script records the actual y/z mesh, material index, selected mode, complex
neff, TE fraction, field overlap with the previous mesh, and optional E/H
fields.  Phase uses only Re(neff); Im(neff) changes are reported as loss/gain.
No result is called complete merely because a mode was found.
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
from lumerical.validation import phase_and_loss_changes, recent_phase_converged  # noqa: E402


def build_fde(app, *, domain_y, domain_z, boundary, pml_layers):
    app.addfde()
    for prop, value in (
        ("solver type", "2D X normal"),
        ("y span", domain_y),
        ("z span", domain_z),
        ("wavelength", 1.55e-6),
        ("define y mesh by", "maximum mesh step"),
        ("define z mesh by", "maximum mesh step"),
        ("dy", 50e-9),
        ("dz", 50e-9),
        ("background material", "SiO2 (Glass) - Palik"),
    ):
        app.set(prop, value)
    for side in ("y min bc", "y max bc", "z min bc", "z max bc"):
        app.set(side, "Metal" if boundary == "metal" else "PML")
    if boundary == "pml":
        app.set("pml layers", int(pml_layers))
    app.addrect()
    for prop, value in (
        ("name", "Si_WG"), ("x", 0), ("y", 0), ("z", 0),
        ("y span", 0.5e-6), ("z span", 0.22e-6),
        ("material", "Si (Silicon) - Palik"),
    ):
        app.set(prop, value)
    app.addmesh()
    mesh = app.get("name")
    for prop, value in (
        ("x", 0), ("x span", 1e-6),
        ("y", 0), ("y span", 1.2e-6),
        ("z", 0), ("z span", 0.8e-6),
        ("override y mesh", True), ("override z mesh", True),
        ("set maximum mesh step", True),
    ):
        app.set(prop, value)
    return app.get("name") or mesh


def _mode_path(name):
    return name if name.startswith("FDE::data::") else f"FDE::data::{name}"


def _neff(app, name):
    value = complex(np.asarray(app.getresult(_mode_path(name), "neff"))[0, 0])
    return value


def _te(app, name):
    return float(np.asarray(
        app.getresult(_mode_path(name), "TE polarization fraction")).ravel()[0])


def _save_mesh(app, outpath):
    app.mesh()
    y = np.asarray(app.getresult("material", "y")).ravel()
    z = np.asarray(app.getresult("material", "z")).ravel()
    index = np.asarray(app.getresult("material", "index_x"))
    np.savez_compressed(outpath, y=y, z=z, index_x=index)
    dy = np.diff(y)
    dz = np.diff(z)
    return {
        "file": outpath.name,
        "cells_y": int(y.size),
        "cells_z": int(z.size),
        "min_dy_nm": float(np.min(np.abs(dy))) * 1e9 if dy.size else None,
        "max_dy_nm": float(np.max(np.abs(dy))) * 1e9 if dy.size else None,
        "min_dz_nm": float(np.min(np.abs(dz))) * 1e9 if dz.size else None,
        "max_dz_nm": float(np.max(np.abs(dz))) * 1e9 if dz.size else None,
    }


def _choose_first_mode(app, found, trial_modes):
    candidates = []
    for i in range(1, min(found, trial_modes) + 1):
        name = f"FDE::data::mode{i}"
        try:
            neff = _neff(app, name)
            te = _te(app, name)
        except Exception:
            continue
        candidates.append({"name": name, "neff": neff, "te": te})
    if not candidates:
        raise RuntimeError("no FDE mode candidates found")
    # A high TE fraction alone can select a cladding/interface mode (the
    # 10 nm smoke test did exactly that).  A guided Si mode must also have a
    # high real neff; choose the highest-neff TE-like candidate, not max TE.
    guided = [c for c in candidates
              if c["te"] >= 0.9 and 1.8 <= c["neff"].real <= 3.5]
    if not guided:
        raise RuntimeError("no guided TE-like FDE mode found")
    selected = max(guided, key=lambda c: c["neff"].real)
    return selected, candidates


def solve_step(app, mesh, step_nm, *, step_index, reference, previous_overlap,
               trial_modes, export_fields, outdir):
    app.switchtolayout()
    app.select(mesh)
    step = float(step_nm) * 1e-9
    app.set("dy", step)
    app.set("dz", step)
    app.select("FDE")
    app.set("wavelength", 1.55e-6)
    app.mesh()
    found = int(app.findmodes())
    if found < 1:
        raise RuntimeError(f"FDE found no mode at core step {step_nm} nm")

    if reference is None:
        selected, candidates = _choose_first_mode(app, found, trial_modes)
        overlap_values = None
    else:
        best = app.bestoverlap(reference)
        selected = {"name": _mode_path(best), "neff": _neff(app, best),
                    "te": _te(app, best)}
        candidates = []
        ov = np.asarray(app.overlap(_mode_path(best), reference), dtype=float)
        overlap_values = ov.ravel().tolist()
        selected["overlap"] = float(ov.ravel()[0])
        selected["power_coupling"] = float(ov.ravel()[1]) if ov.size > 1 else None

    ref_name = f"ref_step_{step_index}"
    app.copydcard(selected["name"], ref_name)

    step_dir = Path(outdir) / f"step_{step_nm:g}nm"
    step_dir.mkdir(parents=True, exist_ok=True)
    mesh_info = _save_mesh(app, step_dir / "mesh_material.npz")
    if export_fields:
        fields = {}
        for kind in ("E", "H"):
            data = app.getresult(selected["name"], kind)
            for key, value in data.items():
                if key == "Lumerical_dataset":
                    continue
                fields[f"{kind}_{key}"] = np.asarray(value)
        np.savez_compressed(step_dir / "target_mode_fields.npz", **fields)

    return {
        "step_nm": float(step_nm),
        "selected_mode": selected["name"],
        "neff": {"real": selected["neff"].real, "imag": selected["neff"].imag},
        "te_fraction": selected["te"],
        "overlap": selected.get("overlap"),
        "power_coupling": selected.get("power_coupling"),
        "overlap_raw": overlap_values,
        "modes_found": int(found),
        "mode_candidates": [
            {"name": c["name"], "neff": {"real": c["neff"].real,
                                          "imag": c["neff"].imag},
             "te_fraction": c["te"]} for c in candidates
        ],
        "mesh": mesh_info,
        "reference_dcard": ref_name,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "progress/07_lumerical/device_fde_local")
    parser.add_argument("--steps-nm", type=float, nargs="+",
                        default=[10.0, 5.0, 2.5])
    parser.add_argument("--domain-um", type=float, nargs=2, default=[3.0, 2.0],
                        metavar=("Y", "Z"))
    parser.add_argument("--boundary", choices=["metal", "pml"], default="metal")
    parser.add_argument("--pml-layers", type=int, default=16)
    parser.add_argument("--trial-modes", type=int, default=10)
    parser.add_argument("--export-fields", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    app = load_lumapi().MODE(hide=True)
    try:
        mesh = build_fde(app, domain_y=args.domain_um[0] * 1e-6,
                         domain_z=args.domain_um[1] * 1e-6,
                         boundary=args.boundary,
                         pml_layers=args.pml_layers)
        app.save(str(args.out / "soi_waveguide_fde_local.lms"))
        runs = []
        reference = None
        previous_overlap = None
        for index, step in enumerate(args.steps_nm):
            run = solve_step(app, mesh, step, step_index=index,
                             reference=reference,
                             previous_overlap=previous_overlap,
                             trial_modes=args.trial_modes,
                             export_fields=args.export_fields,
                             outdir=args.out)
            runs.append(run)
            reference = run["reference_dcard"]
            if run["overlap"] is not None:
                previous_overlap = run["overlap"]
    finally:
        app.close()

    length = 100e-6
    wavelength = 1.55e-6
    checks = []
    for previous, current in zip(runs, runs[1:]):
        check = phase_and_loss_changes(
            previous["neff"]["real"] + 1j * previous["neff"]["imag"],
            current["neff"]["real"] + 1j * current["neff"]["imag"],
            length=length, wavelength=wavelength, phase_limit=0.01)
        checks.append({
            "from_step_nm": previous["step_nm"],
            "to_step_nm": current["step_nm"],
            **check,
        })
    verdict = recent_phase_converged(checks, run_count=len(runs))
    overlaps = [run["overlap"] for run in runs if run["overlap"] is not None]
    mode_identity = bool(overlaps and all(v >= 0.99 for v in overlaps))
    any_imag = any(abs(run["neff"]["imag"]) > 0 for run in runs)
    payload = {
        "stage": "device_fde_local_convergence",
        "timestamp": timestamp(),
        "lumerical": app_version(),
        "geometry": {"si_width_y_um": 0.5, "si_thickness_z_um": 0.22,
                     "waveguide_length_um": 100.0, "uniform_along_x": True},
        "solver": "MODE FDE 2D X normal, full-vector, explicit boundary",
        "boundary": args.boundary,
        "pml_layers": args.pml_layers if args.boundary == "pml" else None,
        "domain_um": {"y": float(args.domain_um[0]), "z": float(args.domain_um[1])},
        "outer_mesh_target_nm": 50.0,
        "core_mesh_um": {"y": 1.2, "z": 0.8},
        "trial_modes": args.trial_modes,
        "steps_nm": [float(s) for s in args.steps_nm],
        "runs": runs,
        "phase_checks": checks,
        "last_two_phase_checks": verdict["last_two"],
        "phase_limit_rad": 0.01,
        "mode_identity_valid": mode_identity,
        "phase_mesh_converged": verdict["passed"],
        "domain_converged": None,
        "boundary_checked": None,
        "loss_check_status": "imag_neff_nonzero" if any_imag else "all_imag_neff_zero",
        "complete": False,
        "complete_scope": "single domain/boundary; domain and boundary scans required",
        "limitation": "Uniform strip only; not a coupler/MZI or wafer cross-section.",
    }
    save_json(args.out / "device_fde_local.json", payload)
    print(json.dumps({
        "boundary": payload["boundary"],
        "domain_um": payload["domain_um"],
        "steps_nm": payload["steps_nm"],
        "neff": [r["neff"]["real"] for r in runs],
        "mode_identity_valid": mode_identity,
        "phase_checks": checks,
        "phase_mesh_converged": verdict["passed"],
        "complete": payload["complete"],
    }, indent=2), flush=True)
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
