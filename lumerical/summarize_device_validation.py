"""Summarize explicit device-validation evidence from a manifest.

No automatic directory globbing: the manifest must name the exact FDE mesh
checkpoints, domain/boundary runs, and EME runs that belong to one comparison.
Missing evidence remains null/false; it is never inferred from a nearby value.
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


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _phase(a, b, length=100e-6, wavelength=1.55e-6):
    if a is None or b is None:
        return None
    return 2.0 * math.pi * length * abs(float(b) - float(a)) / wavelength


def _fde_neff(directory, step_nm=None):
    path = Path(directory)
    json_path = path / "device_fde_local.json"
    if json_path.exists():
        data = _read(json_path)
        runs = data.get("runs", [])
        if not runs:
            return None, data
        if step_nm is None:
            chosen = min(runs, key=lambda r: r["step_nm"])
        else:
            matches = [r for r in runs
                       if abs(float(r["step_nm"]) - float(step_nm)) <= 1e-12]
            if not matches:
                return None, data
            chosen = matches[0]
        return float(chosen["neff"]["real"]), data
    checkpoints = sorted(path.glob("step_*/checkpoint.json"),
                         key=lambda p: float(p.parent.name.split("_")[1].rstrip("nm")))
    if not checkpoints:
        return None, None
    for checkpoint in checkpoints:
        data = _read(checkpoint)
        if step_nm is None or abs(float(data["step_nm"]) - float(step_nm)) <= 1e-12:
            return float(data["neff"]["real"]), data
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = _read(args.manifest)
    out = {
        "manifest": str(args.manifest),
        "fde_mesh_phase_passed": False,
        "fde_domain_checked": False,
        "fde_boundary_checked": False,
        "uniform_strip_fde_complete": False,
        "eme_phase_mapping_passed": False,
        "eme_strict_raw_passive": None,
        "eme_device_complete": False,
        "slab_diagnostic_complete": False,
        "complete_scope": "manifest-defined uniform-strip numerical checks only",
        "complete": False,
        "evidence": {},
        "unmeasured_or_failed": [],
    }

    mesh_dir = Path(manifest["fde_mesh_dir"])
    expected = [float(x) for x in manifest.get("fde_expected_steps_nm", [])]
    checkpoints = sorted(mesh_dir.glob("step_*/checkpoint.json"),
                         key=lambda p: -float(p.parent.name.split("_")[1].rstrip("nm")))
    steps = [float(_read(p)["step_nm"]) for p in checkpoints]
    out["evidence"]["fde_mesh_checkpoints"] = [str(p) for p in checkpoints]
    out["fde_mesh_steps_nm"] = steps
    if len(steps) != len(expected) or any(abs(a - b) > 1e-12
                                          for a, b in zip(steps, expected)):
        out["unmeasured_or_failed"].append("FDE expected mesh steps incomplete")
    else:
        checks = []
        for a, b in zip(checkpoints, checkpoints[1:]):
            da, db = _read(a), _read(b)
            checks.append({
                "from_step_nm": da["step_nm"], "to_step_nm": db["step_nm"],
                "phase_delta_rad": _phase(da["neff"]["real"], db["neff"]["real"]),
                "phase_limit_rad": 0.01,
                "phase_converged": _phase(da["neff"]["real"],
                                           db["neff"]["real"]) <= 0.01,
            })
        out["fde_mesh_phase_checks"] = checks
        out["fde_mesh_phase_passed"] = bool(
            len(checks) >= 2 and all(c["phase_converged"] for c in checks[-2:]))
    overlaps = [_read(p).get("overlap") for p in checkpoints[1:]]
    out["fde_mode_identity_valid"] = bool(
        overlaps and all(v is not None and np.isfinite(v) and v >= 0.99
                         for v in overlaps))

    ref_step = manifest.get("fde_reference_step_nm")
    domain_dirs = [Path(p) for p in manifest.get("fde_domain_dirs", [])]
    if domain_dirs:
        vals = [_fde_neff(p, ref_step)[0] for p in domain_dirs]
        out["fde_domain_values"] = vals
        deltas = ([_phase(a, b) for a, b in zip(vals, vals[1:])]
                  if len(vals) >= 2 and all(v is not None for v in vals) else [])
        out["fde_domain_phase_deltas"] = deltas
        out["fde_domain_checked"] = bool(
            len(vals) >= 3 and all(v is not None for v in vals)
            and len(deltas) >= 2 and all(d is not None and d <= 0.01 for d in deltas))
    boundary = manifest.get("fde_boundary_dirs", {})
    if boundary:
        values = {}
        for name, directory in boundary.items():
            values[name] = _fde_neff(directory, ref_step)[0]
        out["fde_boundary_values"] = values
        names = list(values)
        boundary_deltas = {
            f"{a}->{b}": _phase(values[a], values[b])
            for a, b in zip(names, names[1:])}
        out["fde_boundary_phase_deltas"] = boundary_deltas
        out["fde_boundary_checked"] = bool(
            len(values) >= 3 and all(v is not None for v in values.values())
            and boundary_deltas
            and all(d is not None and d <= 0.01 for d in boundary_deltas.values()))

    eme_dir = Path(manifest["eme_dir"])
    eme_runs = sorted(eme_dir.glob("eme_L*um.json"),
                      key=lambda p: float(_read(p)["length_um"]))
    out["evidence"]["eme_runs"] = [str(p) for p in eme_runs]
    lengths = [float(_read(p)["length_um"]) for p in eme_runs]
    expected_lengths = [float(x) for x in manifest.get(
        "eme_expected_lengths_um", [10.0, 10.1, 10.2])]
    out["eme_lengths_completed"] = lengths
    out["eme_expected_lengths_um"] = expected_lengths
    out["eme_lengths_match_manifest"] = bool(
        len(lengths) == len(expected_lengths)
        and all(abs(a - b) <= 1e-12 for a, b in zip(lengths, expected_lengths)))
    out["eme_flow_complete"] = bool(
        out["eme_lengths_match_manifest"]
        and all(_read(p).get("status") == "nonzero_s_observed" for p in eme_runs))
    out["eme_phase_checks"] = []
    out["eme_phase_checks_complete"] = False
    if eme_runs:
        out["eme_strict_raw_passive"] = all(
            bool(_read(p).get("strict_raw_passive")) for p in eme_runs)
        fde_ref = manifest.get("fde_reference_dir")
        fde_step = manifest.get("fde_reference_step_nm")
        fde_neff = _fde_neff(fde_ref, fde_step)[0] if fde_ref else None
        phase_values = []
        for pa, pb in zip(eme_runs, eme_runs[1:]):
            a, b = _read(pa), _read(pb)
            check = {"from_length_um": a["length_um"],
                     "to_length_um": b["length_um"],
                     "phase_computable": False}
            try:
                s = np.asarray(a["user_s_matrix"]["real"], dtype=float) + \
                    1j * np.asarray(a["user_s_matrix"]["imag"], dtype=float)
                s2 = np.asarray(b["user_s_matrix"]["real"], dtype=float) + \
                     1j * np.asarray(b["user_s_matrix"]["imag"], dtype=float)
                if s.shape != (2, 2) or s2.shape != (2, 2):
                    raise ValueError("user S is not 2x2")
                sa, sb = complex(s[1, 0]), complex(s2[1, 0])
                if not (np.isfinite(sa) and np.isfinite(sb)):
                    raise ValueError("S21 is non-finite")
                if abs(sa) <= 1e-12 or abs(sb) <= 1e-12:
                    raise ValueError("S21 below phase floor")
                delta_l = (float(b["length_um"]) - float(a["length_um"])) * 1e-6
                if delta_l <= 0:
                    raise ValueError("length increment is not positive")
                ratio = sb / sa
                beta = math.atan2(ratio.imag, ratio.real) / delta_l
                eme_neff = beta * 1.55e-6 / (2.0 * math.pi)
                check.update({
                    "phase_computable": True,
                    "s21_from": {"real": sa.real, "imag": sa.imag},
                    "s21_to": {"real": sb.real, "imag": sb.imag},
                    "neff_from_length": float(eme_neff),
                    "fde_neff": fde_neff,
                    "phase_vs_fde_100um_rad": _phase(fde_neff, eme_neff),
                })
                check["phase_passed"] = bool(
                    check["phase_vs_fde_100um_rad"] is not None
                    and check["phase_vs_fde_100um_rad"] <= 0.01)
                if check["phase_vs_fde_100um_rad"] is not None:
                    phase_values.append(check["phase_vs_fde_100um_rad"])
            except Exception as exc:
                check["reason"] = str(exc)
            out["eme_phase_checks"].append(check)
        out["eme_phase_checks_complete"] = bool(
            len(out["eme_phase_checks"]) == max(len(expected_lengths) - 1, 0)
            and all(c.get("phase_computable") for c in out["eme_phase_checks"]))
        out["eme_phase_mapping_passed"] = bool(
            out["eme_flow_complete"] and out["eme_phase_checks_complete"]
            and all(c.get("phase_passed") for c in out["eme_phase_checks"]))
        out["eme_evidence_complete"] = bool(
            eme_runs and all(
                _read(p).get("mesh_matching_valid")
                and _read(p).get("cell_neff_export", {}).get("status") == "saved"
                and all(_read(p).get("port_exports", {}).get(port, {}).get("profile")
                        == "saved" for port in ("port_1", "port_2"))
                for p in eme_runs))
        out["eme_device_complete"] = bool(
            out["eme_phase_mapping_passed"] and
            out["eme_strict_raw_passive"] is True and
            out["eme_evidence_complete"])
    if not out["fde_mesh_phase_passed"]:
        out["unmeasured_or_failed"].append("FDE last two continuous mesh intervals not both <=0.01 rad")
    if not out["eme_phase_mapping_passed"]:
        out["unmeasured_or_failed"].append("EME all declared length intervals not both computable and <=0.01 rad")
    if not out["eme_flow_complete"]:
        out["unmeasured_or_failed"].append("EME expected lengths incomplete")

    out["uniform_strip_fde_complete"] = bool(
        out["fde_mesh_phase_passed"]
        and out["fde_domain_checked"]
        and out["fde_boundary_checked"]
        and out.get("fde_mode_identity_valid") is True)
    out["complete"] = bool(out["uniform_strip_fde_complete"]
                           and out.get("eme_device_complete") is True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(out, indent=2, allow_nan=False))
    return 0 if out["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
