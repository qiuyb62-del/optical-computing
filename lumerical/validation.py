"""Shared acceptance helpers for Lumerical experiments.

The helpers deliberately distinguish "not measured" from "measured zero".
Callers must use real datasets for zero results (or an independent power
measurement); a missing ONA transmission is never converted to 0 here.
"""
from __future__ import annotations

import math

import numpy as np

RTOL = 1e-6
ATOL = 1e-10
ZERO_RELATIVE = 1e-12


def _complex_array(value, name):
    try:
        array = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a complex array") from exc
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def compare_complex(actual, reference, *, scale=None, rtol=RTOL, atol=ATOL):
    """Compare complex fields without global phase alignment.

    ``scale`` is the physical scale B used by the repository's near-zero rule
    (for example ||W||_F ||x||_2).  A zero or near-zero reference is checked
    with an absolute tolerance; otherwise the full complex relative error is
    used.  Non-finite values fail instead of being silently accepted.
    """
    actual = _complex_array(actual, "actual")
    reference = _complex_array(reference, "reference")
    if actual.shape != reference.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {reference.shape}")
    finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)))
    if not finite:
        return {
            "criterion": "invalid",
            "relative_error": None,
            "absolute_error": None,
            "limit": None,
            "passed": False,
            "reason": "non-finite complex field",
        }
    norm_reference = float(np.linalg.norm(reference))
    base = max(1.0, float(scale) if scale is not None else norm_reference)
    absolute = float(np.max(np.abs(actual - reference))) if actual.size else 0.0
    if norm_reference == 0.0 or norm_reference <= ZERO_RELATIVE * base:
        limit = atol * base
        return {
            "criterion": "absolute",
            "relative_error": None,
            "absolute_error": absolute,
            "limit": float(limit),
            "scale": float(base),
            "passed": bool(math.isfinite(absolute) and absolute <= limit),
        }
    relative = float(np.linalg.norm(actual - reference) / norm_reference)
    return {
        "criterion": "relative",
        "relative_error": relative,
        "absolute_error": absolute,
        "limit": float(rtol),
        "scale": float(base),
        "passed": bool(math.isfinite(relative) and relative <= rtol),
    }


def phase_and_loss_changes(neff_a, neff_b, *, length, wavelength,
                           phase_limit=0.01):
    """Separate propagation phase (Re neff) from attenuation (Im neff).

    The phase metric deliberately uses only the real part.  A changing
    imaginary part is reported as a loss/gain change, not as phase error.
    """
    a, b = complex(neff_a), complex(neff_b)
    finite = all(math.isfinite(v) for v in
                 (a.real, a.imag, b.real, b.imag, length, wavelength))
    if not finite or length <= 0 or wavelength <= 0:
        return {
            "finite": False,
            "delta_neff": None,
            "delta_neff_real": None,
            "delta_neff_imag": None,
            "phase_delta_rad": None,
            "power_loss_delta_db": None,
            "phase_limit_rad": float(phase_limit),
            "s21_delta_abs": None,
            "phase_converged": False,
        }
    k0 = 2.0 * math.pi / wavelength
    delta_real = abs(b.real - a.real)
    delta_imag = abs(b.imag - a.imag)
    phase_delta = k0 * length * delta_real
    loss_delta_db = (20.0 / math.log(10.0)) * k0 * length * delta_imag
    s21_a = complex(np.exp(1j * k0 * a * length))
    s21_b = complex(np.exp(1j * k0 * b * length))
    return {
        "finite": True,
        "delta_neff": abs(b - a),
        "delta_neff_real": delta_real,
        "delta_neff_imag": delta_imag,
        "phase_delta_rad": float(phase_delta),
        "power_loss_delta_db": float(loss_delta_db),
        "phase_limit_rad": float(phase_limit),
        "s21_delta_abs": float(abs(s21_a - s21_b)),
        "phase_converged": bool(phase_delta <= phase_limit),
    }


def fde_phase_metrics(neff_a, neff_b, *, length, wavelength, phase_limit=0.01):
    """Backward-compatible wrapper with real-part phase semantics."""
    return phase_and_loss_changes(neff_a, neff_b, length=length,
                                  wavelength=wavelength,
                                  phase_limit=phase_limit)


def strict_passivity(values, *, upper=1.0):
    """Pointwise passive check without clipping or absolute values.

    ``upper`` is intentionally exactly 1.0 for the slab diagnostic.  A future
    numerical tolerance must be justified before the run, not after seeing T.
    """
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        return {"passed": False, "reason": "empty", "max": None, "min": None}
    finite = bool(np.all(np.isfinite(array)))
    maximum = float(np.max(array)) if finite else None
    minimum = float(np.min(array)) if finite else None
    passed = bool(finite and np.all(array >= 0.0) and np.all(array <= upper))
    return {
        "passed": passed,
        "reason": "ok" if passed else "non-finite, negative, or above upper bound",
        "max": maximum,
        "min": minimum,
        "upper": float(upper),
        "count": int(array.size),
    }


def power_ratio_mesh_converged(runs, meshes, *, threshold=1e-3, minimum_meshes=3):
    """Check at least three mesh levels and every adjacent power-ratio pair."""
    meshes = [int(m) for m in meshes]
    if len(meshes) < minimum_meshes:
        return {
            "passed": False,
            "reason": f"need at least {minimum_meshes} mesh levels",
            "pairwise": [],
        }
    pairwise = []
    for a, b in zip(meshes, meshes[1:]):
        ta = np.asarray(runs[str(a)]["T_net_ratio"], dtype=float)
        tb = np.asarray(runs[str(b)]["T_net_ratio"], dtype=float)
        if ta.shape != tb.shape:
            return {"passed": False, "reason": "mesh arrays have different shapes",
                    "pairwise": pairwise}
        delta = float(np.max(np.abs(tb - ta)))
        pairwise.append({"from": a, "to": b, "max_abs_delta_T": delta})
    passed = bool(pairwise and all(item["max_abs_delta_T"] <= threshold
                                   for item in pairwise))
    return {"passed": passed, "reason": "ok" if passed else "delta above threshold",
            "pairwise": pairwise}


def recent_phase_converged(phase_checks, *, run_count, recent=2, minimum_runs=3):
    """Judge only the most recent continuous refinement pairs.

    Historical coarse pairs remain in the record; they must not make the
    terminal verdict permanently false after finer levels have stabilized.
    """
    checks = list(phase_checks)
    if run_count < minimum_runs or len(checks) < recent:
        return {"passed": False, "last_two": checks[-recent:] if checks else [],
                "reason": f"need at least {minimum_runs} runs and {recent} recent checks"}
    last = checks[-recent:]
    passed = bool(all(item.get("phase_converged") for item in last))
    return {"passed": passed, "last_two": last,
            "reason": "ok" if passed else "recent phase check above limit"}


def time_domain_passed(frequency_passed, record_checks):
    """Overall time-domain acceptance: both frequency and every record must pass."""
    if not frequency_passed:
        return False
    if not record_checks:
        return False
    for check in record_checks:
        if not isinstance(check, dict) or not check:
            return False
        if not all(bool(value.get("passed")) for value in check.values()):
            return False
    return True
