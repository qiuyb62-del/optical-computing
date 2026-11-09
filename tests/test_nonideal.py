"""物理映射、可复现误差、静态复用、被动性及独立解析案例。"""
from dataclasses import FrozenInstanceError
import json
import numpy as np
import pytest
from src.mzi import mzi_matrix
from src.svd_matmul import compile_svd, propagate_svd
from src.nonideal import (NonidealConfig, coupler_matrix, physical_mzi_matrix,
                          quantize_phase, realize_nonideal, trace_nonideal,
                          propagate_nonideal, reconstruct_nonideal)


@pytest.mark.parametrize("theta,phi", [(0,0), (np.pi/2,0), (-.7,1.1), (np.pi,2.2),
                                       (5.4,-.5), (-2*np.pi-.2,9.2)])
def test_physical_mapping_keeps_full_phase(theta, phi):
    actual = physical_mzi_matrix([phi, 2*theta, -theta, -theta])
    np.testing.assert_allclose(actual, mzi_matrix(theta, phi), atol=2e-15, rtol=2e-15)


def test_missing_common_phase_is_detectable():
    theta, phi = .8, -.2
    wrong = physical_mzi_matrix([phi, 2*theta, 0, 0])
    np.testing.assert_allclose(wrong, np.exp(1j*theta)*mzi_matrix(theta, phi), atol=1e-15)
    assert np.linalg.norm(wrong-mzi_matrix(theta, phi)) > .1


@pytest.mark.parametrize("eta", [0, .45, .5, .73, 1])
def test_lossless_couplers_remain_unitary(eta):
    B = coupler_matrix(eta)
    np.testing.assert_allclose(B.conj().T @ B, np.eye(2), atol=1e-15)
    T = physical_mzi_matrix([.3, -.6, .8, .9], coupler_through=eta)
    np.testing.assert_allclose(T.conj().T @ T, np.eye(2), atol=1e-15)


def test_loss_db_changes_field_and_power_correctly():
    T = physical_mzi_matrix([.4, .7, -.35, -.35], loss_db=6.0)
    expected = 10**(-6/20) * mzi_matrix(.35, .4)
    np.testing.assert_allclose(T, expected, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(T[:,0])**2, 10**(-6/10), atol=1e-15)


def test_unbalanced_arm_is_not_uniform_loss():
    # θ=0, φ=0: B† diag(a,1) B has known diagonal/off-diagonal entries.
    a = 10**(-3/20)
    actual = physical_mzi_matrix([0,0,0,0], upper_arm_loss_db=3)
    expected = np.array([[(a+1)/2, 1j*(a-1)/2],[-1j*(a-1)/2,(a+1)/2]])
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert np.max(np.linalg.svd(actual,compute_uv=False)) <= 1+1e-15


def test_phase_quantization_boundary_and_ties():
    step = 2*np.pi/16
    assert quantize_phase(-.1*step,4) == 0
    assert quantize_phase(2*np.pi-.1*step,4) == 0
    assert quantize_phase(.5*step,4) == 0
    assert quantize_phase(1.5*step,4) == 2*step
    assert quantize_phase(-.3,None) == -.3
    for value in np.linspace(-12,12,103):
        code = quantize_phase(value,6)
        assert 0 <= code < 2*np.pi
        residual = np.angle(np.exp(1j*(code-value)))
        assert abs(residual) <= np.pi/64+1e-14


@pytest.mark.parametrize("n", [1,2,3,4,8])
def test_zero_error_recovers_ideal_and_does_not_mutate(n):
    rng = np.random.default_rng(42+n)
    W = rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    x = rng.normal(size=n)+1j*rng.normal(size=n)
    saved_W, saved_x = W.copy(), x.copy()
    ideal = compile_svd(W)
    device = realize_nonideal(ideal, seed=12)
    np.testing.assert_allclose(reconstruct_nonideal(device), W, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(propagate_nonideal(x,device), propagate_svd(x,ideal),atol=1e-12,rtol=1e-12)
    np.testing.assert_array_equal(W,saved_W)
    np.testing.assert_array_equal(x,saved_x)


def test_two_channel_loss_has_analytic_circuit_gain():
    W=np.array([[1,2j],[-.4,3]],complex)
    device=realize_nonideal(compile_svd(W),NonidealConfig(loss_db_per_mzi=.7))
    # Both channels encounter two MZIs. This is NOT generally a global loss at N>2.
    np.testing.assert_allclose(reconstruct_nonideal(device), 10**(-1.4/20)*W, atol=1e-13)


def test_static_realizations_seed_and_linearity():
    W=np.array([[1,2j],[3,4]],complex)
    config=NonidealConfig(phase_std_rad=.1,phase_bits=6,loss_db_per_mzi=.2)
    a=realize_nonideal(compile_svd(W),config,seed=123)
    b=realize_nonideal(compile_svd(W),config,seed=123)
    c=realize_nonideal(compile_svd(W),config,seed=124)
    assert a == b
    assert a != c
    x,z=np.array([.2,.7j]),np.array([2j,-1])
    y=propagate_nonideal(x,a)
    np.testing.assert_array_equal(y,propagate_nonideal(x,a))
    np.testing.assert_allclose(propagate_nonideal(x+z,a),y+propagate_nonideal(z,a),atol=1e-13)
    with pytest.raises(FrozenInstanceError):
        a.seed=2
    json.dumps(a.to_dict(),allow_nan=False)


def test_sampling_does_not_change_global_rng():
    np.random.seed(909)
    state=np.random.get_state()
    realize_nonideal(compile_svd(np.eye(2)),NonidealConfig(phase_std_rad=.2),seed=5)
    after=np.random.get_state()
    assert state[0]==after[0]
    np.testing.assert_array_equal(state[1],after[1])
    assert state[2:]==after[2:]


def test_quantization_on_physical_controls_and_noise_after_quantization():
    base=compile_svd(np.array([[1,1j],[.3,-2]],complex))
    clean=realize_nonideal(base,NonidealConfig(phase_bits=4),seed=21)
    noisy=realize_nonideal(base,NonidealConfig(phase_bits=4,phase_std_rad=.01),seed=21)
    for x,y in zip(clean.input_mesh.gates,noisy.input_mesh.gates):
        assert x.commanded_phases == y.commanded_phases
        np.testing.assert_allclose(np.asarray(x.commanded_phases)/(2*np.pi/16),
                                   np.round(np.asarray(x.commanded_phases)/(2*np.pi/16)),atol=1e-14)
        assert x.actual_phases == x.commanded_phases
        assert y.actual_phases != y.commanded_phases
    assert np.linalg.norm(reconstruct_nonideal(clean)-np.array([[1,1j],[.3,-2]])) > 1e-3


@pytest.mark.parametrize("config", [NonidealConfig(phase_std_rad=.2),
    NonidealConfig(phase_bits=4), NonidealConfig(coupler_through=.45),
    NonidealConfig(loss_db_per_mzi=.3), NonidealConfig(upper_arm_loss_db=.5),
    NonidealConfig(phase_std_rad=.1,loss_db_per_mzi=.2,phase_bits=6,coupler_through=.47)])
def test_all_models_are_passive(config):
    rng=np.random.default_rng(8)
    W=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    device=realize_nonideal(compile_svd(W),config,seed=2)
    H=reconstruct_nonideal(device,restore_scale=False)
    assert np.linalg.svd(H,compute_uv=False)[0] <= 1+1e-12
    x=rng.normal(size=4)+1j*rng.normal(size=4)
    tr=trace_nonideal(x,device)
    powers=[np.linalg.norm(v)**2 for v in [x,tr.after_input_mesh,tr.after_attenuation,tr.optical_output]]
    assert np.all(np.diff(powers)<=1e-12*powers[0])
    np.testing.assert_array_equal(propagate_nonideal(np.zeros(4),device),np.zeros(4))


def test_zero_and_rank_deficient_targets():
    for W in [np.zeros((4,4)),np.diag([0,1,2,3])]:
        device=realize_nonideal(compile_svd(W),NonidealConfig(phase_std_rad=.1),seed=1)
        H=reconstruct_nonideal(device)
        assert np.all(np.isfinite(H))
        if not np.any(W):
            np.testing.assert_array_equal(H,W)


@pytest.mark.parametrize("kwargs", [dict(phase_std_rad=-1),dict(phase_std_rad=True),
    dict(phase_std_rad=np.nan),dict(loss_db_per_mzi=-.1),dict(loss_db_per_mzi=np.inf),
    dict(upper_arm_loss_db=-1),dict(phase_bits=0),dict(phase_bits=33),dict(phase_bits=4.0),
    dict(phase_bits=True),dict(coupler_through=-.1),dict(coupler_through=1.1),
    dict(coupler_through=1j)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        NonidealConfig(**kwargs)


@pytest.mark.parametrize("seed", [-1,True,.2])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        realize_nonideal(compile_svd(np.eye(2)),seed=seed)


def test_invalid_inputs():
    circuit=compile_svd(np.eye(2))
    with pytest.raises(ValueError): realize_nonideal(None)
    with pytest.raises(ValueError): realize_nonideal(circuit,{})
    with pytest.raises(ValueError): physical_mzi_matrix([0,0])
    with pytest.raises(ValueError): physical_mzi_matrix([0,0,0,np.nan])
    with pytest.raises(ValueError): reconstruct_nonideal(circuit)
    device=realize_nonideal(circuit)
    for x in [[1],[[1,2]],[1,np.nan]]:
        with pytest.raises(ValueError): propagate_nonideal(x,device)
