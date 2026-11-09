"""Frozen 4-4-3 classifier. Only the first weight multiplication is optical.

Weights follow torch.nn.Linear: W[out, in]; sample x is a column in the circuit.
Bias, ReLU and the 3-class output head remain digital. The readout takes Re(E)
in a fixed phase reference; this assumes coherent field access, not a detector
measuring |E|^2. No per-sample phase alignment or error calibration is applied.
This module needs only NumPy, including when replaying a saved trained model.
"""
from dataclasses import dataclass
import numpy as np
from .ai_data import features, labels
from .svd_matmul import SVDCircuit, trace_svd
from .nonideal import NonidealCircuit, trace_nonideal


@dataclass(frozen=True)
class ClassifierWeights:
    W1: tuple
    b1: tuple
    W2: tuple
    b2: tuple

    def __post_init__(self):
        for name, shape in (("W1", (4, 4)), ("b1", (4,)), ("W2", (3, 4)), ("b2", (3,))):
            a = np.asarray(getattr(self, name))
            if a.shape != shape or np.iscomplexobj(a) or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} must be a finite real array of shape {shape}")
            a = a.astype(np.float64)
            value = tuple(map(float, a)) if a.ndim == 1 else tuple(tuple(map(float, row)) for row in a)
            object.__setattr__(self, name, value)

    def to_dict(self):
        return {name: getattr(self, name) for name in ("W1", "b1", "W2", "b2")}


def digital_forward(X, weights):
    X = features(X)
    linear = X @ np.asarray(weights.W1).T
    hidden = np.maximum(linear + np.asarray(weights.b1), 0.0)
    logits = hidden @ np.asarray(weights.W2).T + np.asarray(weights.b2)
    return {"linear": linear, "hidden": hidden, "logits": logits}


def optical_forward(X, weights, circuit):
    X = features(X)
    if not isinstance(circuit, (SVDCircuit, NonidealCircuit)) or circuit.size != 4:
        raise ValueError("the Iris optical layer requires a compiled 4-channel circuit")
    trace = trace_svd if isinstance(circuit, SVDCircuit) else trace_nonideal
    records = [trace(x, circuit) for x in X]
    linear = np.stack([r.restored_output for r in records])
    raw = np.stack([r.optical_output for r in records])
    # Coherent real-quadrature readout; never magnitude or intensity.
    hidden = np.maximum(linear.real + np.asarray(weights.b1), 0.0)
    logits = hidden @ np.asarray(weights.W2).T + np.asarray(weights.b2)
    powers = np.array([[np.vdot(v, v).real for v in
                        (x, r.after_input_mesh, r.after_attenuation, r.optical_output)]
                       for x, r in zip(X, records)])
    allowance = 1e-10 * powers[:, 0]
    passive = bool(np.all(np.isfinite(powers)) and np.all(np.diff(powers, axis=1) <= allowance[:, None]))
    return {"linear": linear, "hidden": hidden, "logits": logits,
            "raw_optical": raw, "powers": powers, "passivity_passed": passive}


def relative_norm(actual, reference):
    norm = float(np.linalg.norm(reference))
    return float(np.linalg.norm(actual - reference) / norm) if norm > 0 else None


def classification_metrics(y, logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 3 or not np.all(np.isfinite(logits)):
        raise ValueError("logits must have finite shape (samples, 3)")
    y = labels(y, len(logits))
    if not len(y):
        raise ValueError("classification needs at least one example")
    pred = logits.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    nll = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]
    return {"accuracy": float(np.mean(pred == y)), "cross_entropy": float(nll.mean()),
            "confusion_matrix": confusion.tolist(), "predictions": pred.tolist()}


def compare_forward(actual, reference):
    """Whole-test-set norm ratios; not a mean of unstable per-sample ratios."""
    norm = float(np.linalg.norm(reference["linear"]))
    return {
        "linear_relative_error": relative_norm(actual["linear"], reference["linear"]),
        "linear_max_absolute_error": float(np.max(np.abs(actual["linear"] - reference["linear"]))),
        "imaginary_leakage_ratio": float(np.linalg.norm(actual["linear"].imag) / norm) if norm else None,
        "hidden_relative_error": relative_norm(actual["hidden"], reference["hidden"]),
        "logit_relative_error": relative_norm(actual["logits"], reference["logits"]),
        "logit_max_absolute_error": float(np.max(np.abs(actual["logits"] - reference["logits"]))),
        "prediction_agreement": float(np.mean(actual["logits"].argmax(1) == reference["logits"].argmax(1))),
    }
