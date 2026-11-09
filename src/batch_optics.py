"""Batch inputs with the SAME per-device MZI propagation, never a dense W.

Rows are samples. Each local 2x2 transformation acts on two columns only.
Batching changes software scheduling, not the optical circuit or noise model.
"""
import numpy as np
from .mzi import mzi_matrix
from .svd_matmul import SVDCircuit, SVDPropagation
from .nonideal import NonidealCircuit
from .ai_data import features


def _mesh(fields, gates, phases):
    result = fields.copy()
    for channel, matrix in gates:
        upper, lower = result[:, channel].copy(), result[:, channel + 1].copy()
        result[:, channel] = matrix[0, 0] * upper + matrix[0, 1] * lower
        result[:, channel + 1] = matrix[1, 0] * upper + matrix[1, 1] * lower
    return result * np.exp(1j * np.asarray(phases))[None, :]


def trace_batch(E, circuit):
    if not isinstance(circuit, (SVDCircuit, NonidealCircuit)):
        raise ValueError("expected an ideal or static nonideal circuit")
    fields = np.array(E, dtype=np.complex128, copy=True)
    if fields.ndim != 2 or fields.shape[0] == 0 or fields.shape[1] != circuit.size or not np.all(np.isfinite(fields)):
        raise ValueError("batch fields must have finite shape (samples, circuit.size)")
    if isinstance(circuit, SVDCircuit):
        first_gates = [(m, mzi_matrix(theta, phi)) for m, theta, phi in circuit.input_ops]
        last_gates = [(m, mzi_matrix(theta, phi)) for m, theta, phi in circuit.output_ops]
        first_phases, last_phases = circuit.input_phases, circuit.output_phases
    else:
        first_gates = [(g.channel, np.asarray(g.matrix)) for g in circuit.input_mesh.gates]
        last_gates = [(g.channel, np.asarray(g.matrix)) for g in circuit.output_mesh.gates]
        first_phases = circuit.input_mesh.actual_output_phases
        last_phases = circuit.output_mesh.actual_output_phases
    try:
        with np.errstate(over="raise", invalid="raise"):
            first = _mesh(fields, first_gates, first_phases)
            attenuated = first * np.asarray(circuit.amplitude_transmissions)[None, :]
            optical = _mesh(attenuated, last_gates, last_phases)
            restored = circuit.recovery_scale * optical
    except FloatingPointError as exc:
        raise ValueError("propagation exceeded the finite complex128 range; rescale W or E") from exc
    return SVDPropagation(first, attenuated, optical, restored)


def optical_forward_batch(X, weights, circuit):
    X = features(X, weights.size)
    if not isinstance(circuit, (SVDCircuit, NonidealCircuit)) or circuit.size != weights.size:
        raise ValueError("classifier and circuit sizes must match")
    result = trace_batch(X, circuit)
    linear = result.restored_output
    hidden = np.maximum(linear.real + np.asarray(weights.b1), 0.0)
    logits = hidden @ np.asarray(weights.W2).T + np.asarray(weights.b2)
    powers = np.column_stack([np.sum(np.abs(v)**2, axis=1) for v in
                              (X, result.after_input_mesh, result.after_attenuation, result.optical_output)])
    passive = bool(np.all(np.isfinite(powers)) and np.all(np.diff(powers, axis=1) <= (1e-10 * powers[:, 0])[:, None]))
    return {"linear": linear, "hidden": hidden, "logits": logits, "raw_optical": result.optical_output,
            "powers": powers, "passivity_passed": passive}
