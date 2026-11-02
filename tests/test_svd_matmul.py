import dataclasses
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose
import src.svd_matmul as svd_module
from src.svd_matmul import compile_svd, trace_svd, propagate_svd, reconstruct_svd
from src.svd_validation import comparison, evaluate


def check_matrix(W, rng):
    W = np.asarray(W, dtype=complex)
    before = W.copy()
    circuit = compile_svd(W)
    n = len(W)
    assert len(circuit.input_ops) == len(circuit.output_ops) == n * (n - 1) // 2
    assert circuit.recovery_scale >= 1
    assert np.all((np.array(circuit.amplitude_transmissions) >= 0)
                  & (np.array(circuit.amplitude_transmissions) <= 1))
    reconstructed = reconstruct_svd(circuit)
    assert comparison(reconstructed, W)["passed"]
    assert_allclose(reconstruct_svd(circuit, restore_scale=False),
                    W / circuit.recovery_scale, atol=1e-12, rtol=1e-12)
    inputs = list(np.eye(n, dtype=complex))
    inputs += [rng.normal(size=n) + 1j * rng.normal(size=n), np.zeros(n)]
    for x in inputs:
        original = x.copy()
        trace = trace_svd(x, circuit)
        assert evaluate(W, x, circuit, trace, reconstructed)["passed"]
        assert trace.restored_output.dtype == np.complex128
        assert_allclose(x, original, atol=0, rtol=0)
        assert not np.shares_memory(trace.restored_output, x)
    assert_allclose(W, before, atol=0, rtol=0)
    json.dumps(circuit.to_dict(), allow_nan=False)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("complex_input", [False, True])
def test_random_general_matrices(n, complex_input):
    rng = np.random.default_rng(20261102)
    for scale in [.01, 1, 100]:
        for _ in range(5):
            W = rng.normal(size=(n, n)).astype(complex)
            if complex_input:
                W += 1j * rng.normal(size=(n, n))
            check_matrix(scale * W, rng)


@pytest.mark.parametrize("W", [
    np.zeros((4, 4)), np.eye(4), np.diag([-4, -.5, 0, 2]),
    np.ones((4, 4)), np.diag([1, 1e-6, 1e-12, 1e-14]),
    np.exp(.731j) * np.eye(4), [[1, 2], [3, 4]], [[1, 1j], [2 - 1j, 0]],
])
def test_special_matrices(W):
    check_matrix(W, np.random.default_rng(17))


def test_known_hand_computed_result_and_scale():
    W, x = np.array([[1, 2], [3, 4]]), np.array([1, -1])
    circuit = compile_svd(W)
    assert circuit.recovery_scale > 1
    assert_allclose(propagate_svd(x, circuit), [-1, -1], atol=1e-12, rtol=0)
    assert_allclose(propagate_svd(x, circuit, restore_scale=False),
                    np.array([-1, -1]) / circuit.recovery_scale, atol=1e-12, rtol=0)


def test_amplitude_transmission_is_not_power_transmission():
    circuit = compile_svd(np.diag([.5, .25]))
    trace = trace_svd([1, 0], circuit)
    assert circuit.recovery_scale == 1
    assert_allclose(trace.restored_output, [.5, 0], atol=1e-12, rtol=0)
    assert_allclose(np.vdot(trace.optical_output, trace.optical_output), .25, atol=1e-12)


def test_zero_matrix_nullspace_and_zero_input():
    zero = compile_svd(np.zeros((2, 2)))
    assert zero.recovery_scale == 1
    assert zero.amplitude_transmissions == (0., 0.)
    assert_allclose(propagate_svd([2, 3j], zero), 0, atol=0, rtol=0)
    W = np.array([[1., 1.], [2., 2.]])
    x = np.array([1., -1.])
    circuit = compile_svd(W)
    metrics = evaluate(W, x, circuit, trace_svd(x, circuit), reconstruct_svd(circuit))
    assert metrics["vector"]["criterion"] == "absolute_near_zero"
    assert metrics["vector"]["relative_error"] is None
    assert metrics["passed"]
    assert_allclose(propagate_svd([0, 0], circuit), 0, atol=0, rtol=0)


def test_near_null_direction_uses_scale_aware_absolute_error():
    W, x = np.diag([1., 1e-14]), np.array([0., 1.])
    circuit = compile_svd(W)
    metrics = evaluate(W, x, circuit, trace_svd(x, circuit), reconstruct_svd(circuit))
    assert metrics["vector"]["criterion"] == "absolute_near_zero"
    assert metrics["vector"]["relative_error"] is not None
    assert metrics["passed"]
    # 普通输出与整体很小的 W 仍使用相对检查，不能被绝对容差掩盖。
    assert comparison([0.], [1e-14], reference_scale=1e-14)["criterion"] == "relative"
    assert not comparison([0.], [1e-14], reference_scale=1e-14)["passed"]
    assert not comparison([1.], [0.])["passed"]
    assert not comparison([np.nan], [0.])["passed"]


def test_compiled_parameters_are_reused_and_propagation_never_reconstructs(monkeypatch):
    W = np.array([[1 + 2j, 3], [-.7j, 2 - 1j]])
    circuit = compile_svd(W)
    def forbidden(*args, **kwargs):
        raise AssertionError("Propagation must only apply stored device parameters")
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr(svd_module, "decompose_clements", forbidden)
    monkeypatch.setattr(svd_module, "reconstruct_mesh", forbidden)
    for x in [np.array([1, 2j]), np.array([-3j, .2]), np.array([0, 0])]:
        assert_allclose(propagate_svd(x, circuit), W @ x, atol=1e-12, rtol=1e-12)
    with pytest.raises(dataclasses.FrozenInstanceError):
        circuit.recovery_scale = 2


@pytest.mark.parametrize("W", [[], [1, 2], [[1, 2, 3], [4, 5, 6]],
                                   [[[1]]], [[np.nan]], [[np.inf]], [["bad"]]])
def test_invalid_matrices(W):
    with pytest.raises(ValueError):
        compile_svd(W)


@pytest.mark.parametrize("x", [[1], [[1, 2]], [1, np.inf], [np.nan, 0]])
def test_invalid_inputs(x):
    with pytest.raises(ValueError):
        propagate_svd(x, compile_svd(np.eye(2)))


def test_invalid_circuit():
    with pytest.raises(ValueError):
        propagate_svd([1, 2], None)
    with pytest.raises(ValueError):
        reconstruct_svd(None)
