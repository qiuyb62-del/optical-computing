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


def build(app, *, length_um, domain_um, cell_modes, boundary, pml_layers):
    length = float(length_um) * 1e-6
    app.addeme()
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
    for side in ("y min bc", "y max bc", "z min bc", "z max bc"):
        app.set(side, "Metal" if boundary == "metal" else "PML")
    if boundary == "pml":
        app.set("pml layers", int(pml_layers))
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
    return ports


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
             outdir, do_run):
    app = load_lumapi().MODE(hide=True)
    result = {
        "length_um": float(length_um),
        "domain_um": {"y": float(domain_um[0]), "z": float(domain_um[1])},
        "cell_modes": int(cell_modes),
        "boundary": boundary,
        "pml_layers": int(pml_layers) if boundary == "pml" else None,
        "status": "layout_only",
        "complete": False,
        "valid_s_parameters": False,
    }
    try:
        result["ports"] = build(app, length_um=length_um, domain_um=domain_um,
                                cell_modes=cell_modes, boundary=boundary,
                                pml_layers=pml_layers)
        app.save(str(Path(outdir) / f"eme_L{float(length_um):g}um.lms"))
        if do_run:
            app.run()
            app.emepropagate()
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "progress/07_lumerical/device_eme_minimal")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--length-um", type=float, default=10.0)
    parser.add_argument("--length-scan", action="store_true",
                        help="run L, L+0.1 and L+0.2 um")
    parser.add_argument("--domain-um", type=float, nargs=2, default=[4.0, 3.0],
                        metavar=("Y", "Z"))
    parser.add_argument("--core-step-nm", type=float, default=None,
                        help="recorded for matching; EME uses one uniform cell")
    parser.add_argument("--cell-modes", type=int, default=5)
    parser.add_argument("--boundary", choices=["metal", "pml"], default="pml")
    parser.add_argument("--pml-layers", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lengths = ([args.length_um, args.length_um + 0.1, args.length_um + 0.2]
               if args.length_scan else [args.length_um])
    runs = []
    for length in lengths:
        runs.append(run_case(length_um=length, domain_um=args.domain_um,
                             cell_modes=args.cell_modes, boundary=args.boundary,
                             pml_layers=args.pml_layers, outdir=args.out,
                             do_run=args.run))
    length_checks = []
    for a, b in zip(runs, runs[1:]):
        if not (a.get("nonzero_s_observed") and b.get("nonzero_s_observed")):
            continue
        sa = complex(a["user_s_matrix"]["real"][1][0], a["user_s_matrix"]["imag"][1][0])
        sb = complex(b["user_s_matrix"]["real"][1][0], b["user_s_matrix"]["imag"][1][0])
        delta_l = (b["length_um"] - a["length_um"]) * 1e-6
        beta = math.atan2((sb / sa).imag, (sb / sa).real) / delta_l
        neff = beta * 1.55e-6 / (2.0 * math.pi)
        length_checks.append({
            "from_length_um": a["length_um"], "to_length_um": b["length_um"],
            "delta_length_um": b["length_um"] - a["length_um"],
            "s21_from": {"real": sa.real, "imag": sa.imag},
            "s21_to": {"real": sb.real, "imag": sb.imag},
            "beta_from_length": float(beta),
            "neff_from_length": float(neff),
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
        "runs": runs,
        "length_checks": length_checks,
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
    return 0 if any(r.get("nonzero_s_observed") for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
