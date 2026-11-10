"""Minimal 3D EME straight-waveguide audit entry.

``addeme()`` creates two default ports; this script configures those ports and
must not add two more.  A nonzero 2x2 user S is only a workflow check.  Raw
passivity, mode identity, mesh/domain convergence and the length-derived phase
are reported separately; none of them is promoted to "complete" automatically.

With ``--length-scan --run`` the script runs L and L+0.1 um (and optionally
L+0.2 um) and estimates beta/neff from the S21 phase ratio.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumerical.interconnect import app_version, load_lumapi, save_json, timestamp  # noqa: E402


def build(app, *, length_um, domain_um, cell_modes, boundary, pml_layers,
          core_step_nm):
    length = float(length_um) * 1e-6
    app.addeme()
    app.select("EME")
    for prop, value in (
        ("solver type", "3D: X Prop"),
        ("wavelength", 1.55e-6),
        ("x min", -length / 2.0),
        ("y", 0.0), ("z", 0.0),
        ("number of modes for all cell groups", int(cell_modes)),
        ("y span", float(domain_um[0]) * 1e-6),
        ("z span", float(domain_um[1]) * 1e-6),
        ("number of cell groups", 1),
        ("cells", 1),
        ("group spans", length),
        ("energy conservation", "none"),
        ("background material", "SiO2 (Glass) - Palik"),
    ):
        app.set(prop, value)
    if boundary == "metal":
        raise ValueError("EME 3D:X in this build does not expose Metal y/z boundaries; use PML")
    # Set solver-level PML while EME is selected, before creating core_mesh.
    app.select("EME")
    app.set("pml layers", int(pml_layers))
    actual_pml_layers = int(app.get("pml layers"))
    if actual_pml_layers != int(pml_layers):
        raise RuntimeError(
            f"EME PML-layer setting did not take effect: "
            f"{actual_pml_layers} != {pml_layers}")
    if core_step_nm is not None:
        if not (np.isfinite(core_step_nm) and core_step_nm > 0):
            raise ValueError("core_step_nm must be finite and positive")
        for prop, value in (
            ("define y mesh by", "maximum mesh step"),
            ("define z mesh by", "maximum mesh step"),
            ("dy", 50e-9), ("dz", 50e-9),
        ):
            app.set(prop, value)
        app.addmesh()
        for prop, value in (
            ("name", "core_mesh"),
            ("x", 0.0), ("x span", length + 4e-6),
            ("y", 0.0), ("y span", 1.2e-6),
            ("z", 0.0), ("z span", 0.8e-6),
            ("override y mesh", True), ("override z mesh", True),
            ("set maximum mesh step", True),
            ("dy", float(core_step_nm) * 1e-9),
            ("dz", float(core_step_nm) * 1e-9),
        ):
            app.set(prop, value)
    app.addrect()
    for prop, value in (
        ("name", "Si_WG"), ("x", 0), ("y", 0), ("z", 0),
        ("x span", length + 4e-6),
        ("y span", 0.5e-6), ("z span", 0.22e-6),
        ("material", "Si (Silicon) - Palik"),
    ):
        app.set(prop, value)
    # addeme() already creates these two ports. Do NOT add two more.
    ports = ["EME::Ports::port_1", "EME::Ports::port_2"]
    for path, location in zip(ports, ("left", "right")):
        app.select(path)
        for prop, value in (
            ("port location", location),
            ("use full simulation span", True),
            ("mode selection", "fundamental TE mode"),
        ):
            app.set(prop, value)
    return {
        "ports": ports,
        "pml_layers_requested": int(pml_layers),
        "pml_layers_actual": actual_pml_layers,
        "boundary_requested": boundary,
        "boundary_applied": "PML",
    }


def _cell_export_payload(cell):
    """Build one unique-key payload for the cell mesh/field NPZ."""
    payload = {
        k: np.asarray(v)
        for k, v in cell.items()
        if k != "Lumerical_dataset"
    }
    payload.update(y=np.asarray(cell["y"]).ravel(),
                   z=np.asarray(cell["z"]).ravel(),
                   index=np.asarray(cell["index"]))
    return payload


def _complex_json(value):
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        return {"real": arr.real.tolist(), "imag": arr.imag.tolist()}
    return arr.tolist()


def _s_matrix_checks(s):
    s = np.asarray(s, dtype=complex)
    finite = bool(np.all(np.isfinite(s)))
    shape_ok = s.shape == (2, 2)
    raw_max = float(np.max(np.abs(s))) if s.size else 0.0
    singular = np.linalg.svd(s, compute_uv=False) if shape_ok and finite else np.array([])
    max_sv = float(np.max(singular)) if singular.size else None
    unitarity = (float(np.max(np.abs(s.conj().T @ s - np.eye(2))))
                 if shape_ok and finite else None)
    nonzero = bool(shape_ok and finite and np.any(np.abs(s) > 1e-12))
    return {
        "user_s_matrix_shape": list(s.shape),
        "user_s_matrix": {"real": s.real.tolist(), "imag": s.imag.tolist()},
        "user_s_matrix_max_abs": raw_max,
        "user_s_matrix_max_singular_value": max_sv,
        "unitarity_error": unitarity,
        "nonzero_s_observed": nonzero,
        "strict_raw_passive": bool(finite and shape_ok and raw_max <= 1.0
                                   and max_sv is not None and max_sv <= 1.0),
        "candidate_lossless": bool(unitarity is not None and unitarity <= 1e-6),
        "valid_s_parameters": False,
        "complete": False,
    }


def run_case(*, length_um, domain_um, cell_modes, boundary, pml_layers,
             core_step_nm, outdir, do_run):
    app = load_lumapi().MODE(hide=True)
    result = {
        "length_um": float(length_um),
        "domain_um": {"y": float(domain_um[0]), "z": float(domain_um[1])},
        "cell_modes": int(cell_modes),
        "boundary": boundary,
        "pml_layers": int(pml_layers) if boundary == "pml" else None,
        "core_step_nm": float(core_step_nm) if core_step_nm is not None else None,
        "mesh_parameter_applied": core_step_nm is not None,
        "mesh_matching_valid": False,
        "mesh_export": {"status": "not_run"},
        "boundary_applied": ("pml_default" if boundary == "pml"
                             else "not_applied_eme_default"),
        "status": "layout_only",
        "complete": False,
        "valid_s_parameters": False,
    }
    try:
        built = build(app, length_um=length_um, domain_um=domain_um,
                      cell_modes=cell_modes, boundary=boundary,
                      pml_layers=pml_layers, core_step_nm=core_step_nm)
        result["ports"] = built["ports"]
        result["pml_layers_requested"] = built["pml_layers_requested"]
        result["pml_layers_actual"] = built["pml_layers_actual"]
        result["boundary_requested"] = built["boundary_requested"]
        result["boundary_applied"] = built["boundary_applied"]
        if core_step_nm is None:
            app.save(str(Path(outdir) / f"eme_L{float(length_um):g}um.lms"))
        else:
            result["native_layout_saved"] = False
            result["native_layout_reason"] = "core mesh makes .lms very large; raw JSON kept"
        if do_run:
            app.run()
            app.emepropagate()
            # EME does not expose the FDE "material" d-card.  In Analysis
            # Mode the cell/port providers do expose mesh coordinates, index,
            # mode fields, neff and overlap.  Export those explicitly.
            try:
                cell = app.getresult("EME::Cells::cell_1", "mode fields")
                y = np.asarray(cell["y"]).ravel()
                z = np.asarray(cell["z"]).ravel()
                index = np.asarray(cell["index"])
                payload = _cell_export_payload(cell)
                np.savez_compressed(
                    Path(outdir) / f"eme_L{float(length_um):g}um_cell_fields.npz",
                    **payload)
                min_dy = float(np.min(np.abs(np.diff(y)))) * 1e9
                min_dz = float(np.min(np.abs(np.diff(z)))) * 1e9
                result["mesh_export"] = {
                    "status": "saved",
                    "source": "EME::Cells::cell_1 mode fields",
                    "cells_y": int(y.size), "cells_z": int(z.size),
                    "min_dy_nm": min_dy, "max_dy_nm": float(np.max(np.abs(np.diff(y)))) * 1e9,
                    "min_dz_nm": min_dz, "max_dz_nm": float(np.max(np.abs(np.diff(z)))) * 1e9,
                }
                if core_step_nm is not None:
                    result["mesh_matching_valid"] = bool(
                        min_dy <= float(core_step_nm) * 1.01
                        and min_dz <= float(core_step_nm) * 1.01)
            except Exception as exc:
                result["mesh_export"] = {"status": "failed", "error": str(exc)}
                result["mesh_matching_valid"] = False

            # A port/profile failure must not erase an already saved mesh.
            result["cell_neff_export"] = {"status": "not_measured"}
            result["port_exports"] = {}
            try:
                result["cell_neff"] = _complex_json(
                    app.getresult("EME::Cells::cell_1", "neff")["neff"])
                result["cell_neff_export"] = {"status": "saved"}
            except Exception as exc:
                result["cell_neff_export"] = {"status": "failed", "error": str(exc)}
            for port in ("port_1", "port_2"):
                provider = f"EME::Ports::{port}"
                port_status = {"neff": "not_measured", "profile": "not_measured",
                               "overlap_forward": "not_measured",
                               "overlap_backward": "not_measured"}
                try:
                    result.setdefault("port_neff", {})[port] = _complex_json(
                        app.getresult(provider, "neff")["neff"])
                    port_status["neff"] = "saved"
                except Exception as exc:
                    port_status["neff"] = f"failed: {exc}"
                try:
                    profile = app.getresult(provider, "mode profiles")
                    np.savez_compressed(
                        Path(outdir) / f"eme_L{float(length_um):g}um_{port}_profile.npz",
                        **{k: np.asarray(v) for k, v in profile.items()
                           if k != "Lumerical_dataset"})
                    port_status["profile"] = "saved"
                except Exception as exc:
                    port_status["profile"] = f"failed: {exc}"
                for direction in ("overlap_forward", "overlap_backward"):
                    try:
                        arr = np.asarray(app.getresult(provider, direction))
                        np.savez_compressed(
                            Path(outdir) / f"eme_L{float(length_um):g}um_{port}_{direction}.npz",
                            overlap=arr)
                        port_status[direction] = "saved"
                    except Exception as exc:
                        port_status[direction] = f"failed: {exc}"
                result["port_exports"][port] = port_status
            s = np.asarray(app.getresult("EME", "user s matrix"), dtype=complex)
            result.update(_s_matrix_checks(s))
            result["status"] = ("nonzero_s_observed"
                                if result["nonzero_s_observed"]
                                else "invalid_shape_or_empty_s")
            result["note"] = ("Candidate raw S only. Mode identity, mesh/domain "
                              "convergence and length phase are separate checks.")
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["valid_s_parameters"] = False
        result["complete"] = False
    finally:
        app.close()
    save_json(Path(outdir) / f"eme_L{float(length_um):g}um.json", result)
    return result


def run_propagation_sweep(*, length_um, domain_um, cell_modes, boundary,
                          pml_layers, core_step_nm, outdir):
    """Solve the cross-section once, then sweep three propagation lengths."""
    app = load_lumapi().MODE(hide=True)
    runs = []
    try:
        print("sweep: build", flush=True)
        built = build(app, length_um=length_um, domain_um=domain_um,
                      cell_modes=cell_modes, boundary=boundary,
                      pml_layers=pml_layers, core_step_nm=core_step_nm)
        print("sweep: run", flush=True)
        app.run()
        print("sweep: propagate", flush=True)
        app.emepropagate()
        print("sweep: evidence", flush=True)
        evidence = {
            "mesh_export": {"status": "not_run"},
            "mesh_parameter_applied": core_step_nm is not None,
            "mesh_matching_valid": False,
            "cell_neff_export": {"status": "not_measured"},
            "port_exports": {},
            "pml_layers_requested": built["pml_layers_requested"],
            "pml_layers_actual": built["pml_layers_actual"],
            "boundary_applied": built["boundary_applied"],
        }
        try:
            cell = app.getresult("EME::Cells::cell_1", "mode fields")
            y = np.asarray(cell["y"]).ravel()
            z = np.asarray(cell["z"]).ravel()
            np.savez_compressed(
                Path(outdir) / "eme_sweep_cell_fields.npz",
                **_cell_export_payload(cell))
            min_dy = float(np.min(np.abs(np.diff(y)))) * 1e9
            min_dz = float(np.min(np.abs(np.diff(z)))) * 1e9
            evidence["mesh_export"] = {
                "status": "saved", "source": "EME::Cells::cell_1 mode fields",
                "cells_y": int(y.size), "cells_z": int(z.size),
                "min_dy_nm": min_dy, "min_dz_nm": min_dz,
                "max_dy_nm": float(np.max(np.abs(np.diff(y)))) * 1e9,
                "max_dz_nm": float(np.max(np.abs(np.diff(z)))) * 1e9,
            }
            if core_step_nm is not None:
                evidence["mesh_matching_valid"] = bool(
                    min_dy <= float(core_step_nm) * 1.01
                    and min_dz <= float(core_step_nm) * 1.01)
        except Exception as exc:
            evidence["mesh_export"] = {"status": "failed", "error": str(exc)}
        try:
            evidence["cell_neff"] = _complex_json(
                app.getresult("EME::Cells::cell_1", "neff")["neff"])
            evidence["cell_neff_export"] = {"status": "saved"}
        except Exception as exc:
            evidence["cell_neff_export"] = {"status": "failed", "error": str(exc)}
        for port in ("port_1", "port_2"):
            provider = f"EME::Ports::{port}"
            status = {}
            try:
                evidence.setdefault("port_neff", {})[port] = _complex_json(
                    app.getresult(provider, "neff")["neff"])
                status["neff"] = "saved"
            except Exception as exc:
                status["neff"] = f"failed: {exc}"
            try:
                profile = app.getresult(provider, "mode profiles")
                np.savez_compressed(
                    Path(outdir) / f"eme_sweep_{port}_profile.npz",
                    **{k: np.asarray(v) for k, v in profile.items()
                       if k != "Lumerical_dataset"})
                status["profile"] = "saved"
            except Exception as exc:
                status["profile"] = f"failed: {exc}"
            evidence["port_exports"][port] = status

        print("sweep: set analysis", flush=True)
        app.setemeanalysis("propagation sweep", 1)
        app.setemeanalysis("parameter", "group span 1")
        app.setemeanalysis("start", float(length_um) * 1e-6)
        app.setemeanalysis("stop", (float(length_um) + 0.2) * 1e-6)
        app.setemeanalysis("number of points", 3)
        print("sweep: emesweep", flush=True)
        app.emesweep()
        print("sweep: read", flush=True)
        scan = app.getemesweep("S")
        lengths = np.asarray(scan["group_span_1"]).ravel() * 1e6
        s11 = np.asarray(scan["s11"]).ravel()
        s12 = np.asarray(scan["s12"]).ravel()
        s21 = np.asarray(scan["s21"]).ravel()
        s22 = np.asarray(scan["s22"]).ravel()
        if not (lengths.size == s11.size == s12.size == s21.size == s22.size == 3):
            raise RuntimeError("propagation sweep did not return three points")
        for i, length in enumerate(lengths):
            s = np.array([[s11[i], s12[i]], [s21[i], s22[i]]], dtype=complex)
            item = {
                "length_um": float(length),
                "domain_um": {"y": float(domain_um[0]), "z": float(domain_um[1])},
                "cell_modes": int(cell_modes),
                "boundary": boundary,
                "pml_layers": int(pml_layers),
                "core_step_nm": float(core_step_nm) if core_step_nm is not None else None,
                "propagation_sweep": True,
                "status": "nonzero_s_observed" if np.any(np.abs(s) > 1e-12)
                          else "invalid_shape_or_empty_s",
                **evidence,
            }
            item.update(_s_matrix_checks(s))
            runs.append(item)
            save_json(Path(outdir) / f"eme_L{float(length):g}um.json", item)
    finally:
        app.close()
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "progress/07_lumerical/device_eme_minimal")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--length-um", type=float, default=10.0)
    parser.add_argument("--length-scan", action="store_true",
                        help="run L, L+0.1 and L+0.2 um")
    parser.add_argument("--propagation-sweep", action="store_true",
                        help="solve once and sweep L..L+0.2 with EME analysis")
    parser.add_argument("--domain-um", type=float, nargs=2, default=[4.0, 3.0],
                        metavar=("Y", "Z"))
    parser.add_argument("--core-step-nm", type=float, default=None,
                        help="recorded for matching; EME uses one uniform cell")
    parser.add_argument("--cell-modes", type=int, default=5)
    parser.add_argument("--boundary", choices=["metal", "pml"], default="pml")
    parser.add_argument("--pml-layers", type=int, default=16)
    args = parser.parse_args()
    if args.boundary == "metal":
        parser.error("EME 3D:X in this build supports PML only; use --boundary pml")
    if args.core_step_nm is not None and not (
            np.isfinite(args.core_step_nm) and args.core_step_nm > 0):
        parser.error("--core-step-nm must be finite and positive")
    if args.cell_modes < 1:
        parser.error("--cell-modes must be a positive integer")
    if args.boundary == "pml" and args.pml_layers < 1:
        parser.error("--pml-layers must be a positive integer")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.propagation_sweep:
        if not args.run:
            parser.error("--propagation-sweep requires --run")
        lengths = [args.length_um, args.length_um + 0.1, args.length_um + 0.2]
        runs = run_propagation_sweep(
            length_um=args.length_um, domain_um=args.domain_um,
            cell_modes=args.cell_modes, boundary=args.boundary,
            pml_layers=args.pml_layers, core_step_nm=args.core_step_nm,
            outdir=args.out)
    else:
        lengths = ([args.length_um, args.length_um + 0.1, args.length_um + 0.2]
                   if args.length_scan else [args.length_um])
        runs = []
        for length in lengths:
            runs.append(run_case(length_um=length, domain_um=args.domain_um,
                                 cell_modes=args.cell_modes, boundary=args.boundary,
                                 pml_layers=args.pml_layers,
                                 core_step_nm=args.core_step_nm, outdir=args.out,
                                 do_run=args.run))
    length_checks = []
    for a, b in zip(runs, runs[1:]):
        if not (a.get("nonzero_s_observed") and b.get("nonzero_s_observed")):
            length_checks.append({
                "from_length_um": a["length_um"], "to_length_um": b["length_um"],
                "phase_computable": False, "reason": "a length run failed",
            })
            continue
        sa = complex(a["user_s_matrix"]["real"][1][0], a["user_s_matrix"]["imag"][1][0])
        sb = complex(b["user_s_matrix"]["real"][1][0], b["user_s_matrix"]["imag"][1][0])
        if min(abs(sa), abs(sb)) <= 1e-12:
            length_checks.append({
                "from_length_um": a["length_um"], "to_length_um": b["length_um"],
                "phase_computable": False, "reason": "S21 below phase floor",
                "abs_s21_from": abs(sa), "abs_s21_to": abs(sb),
            })
            continue
        delta_l = (b["length_um"] - a["length_um"]) * 1e-6
        beta = math.atan2((sb / sa).imag, (sb / sa).real) / delta_l
        neff = beta * 1.55e-6 / (2.0 * math.pi)
        length_checks.append({
            "from_length_um": a["length_um"], "to_length_um": b["length_um"],
            "delta_length_um": b["length_um"] - a["length_um"],
            "s21_from": {"real": sa.real, "imag": sa.imag},
            "s21_to": {"real": sb.real, "imag": sb.imag},
            "abs_s21_from": abs(sa), "abs_s21_to": abs(sb),
            "beta_from_length": float(beta),
            "neff_from_length": float(neff),
            "phase_computable": True,
            "phase_converged": None,
        })
    payload = {
        "stage": "device_eme_minimal",
        "timestamp": timestamp(),
        "lumerical": app_version(),
        "solver": "MODE EME 3D: X Prop, one uniform cell",
        "domain_um": {"y": float(args.domain_um[0]), "z": float(args.domain_um[1])},
        "cell_modes": int(args.cell_modes),
        "boundary": args.boundary,
        "pml_layers": int(args.pml_layers) if args.boundary == "pml" else None,
        "core_step_nm": args.core_step_nm,
        "lengths_um": [float(x) for x in lengths],
        "expected_runs": len(lengths),
        "completed_runs": sum(1 for r in runs if r.get("status") == "nonzero_s_observed"),
        "failed_runs": [r["length_um"] for r in runs if r.get("status") == "error"],
        "expected_length_checks": max(len(lengths) - 1, 0),
        "computed_length_checks": sum(1 for c in length_checks
                                      if c.get("phase_computable")),
        "runs": runs,
        "length_checks": length_checks,
        "flow_complete": bool(all(r.get("status") == "nonzero_s_observed" for r in runs)
                              and len(length_checks) == max(len(lengths) - 1, 0)),
        "complete": False,
        "complete_scope": "candidate S only; passivity/mode identity/convergence not accepted",
        "note": "Do not use a nonzero 2x2 S as device acceptance.",
    }
    save_json(args.out / "eme_minimal.json", payload)
    print(json.dumps({
        "statuses": [r["status"] for r in runs],
        "lengths_um": payload["lengths_um"],
        "length_checks": length_checks,
        "complete": payload["complete"],
    }, indent=2), flush=True)
    if not args.run:
        return 0 if all(r.get("status") == "layout_only" for r in runs) else 1
    return 0 if payload["flow_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
