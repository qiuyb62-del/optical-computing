import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.mzi import mzi, mzi_matrix, apply_mzi


def close(actual, expected):
    assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_identity_and_phase():
    close(mzi_matrix(0, 0), np.eye(2))
    close(mzi_matrix(0, np.pi/2), np.diag([1j, 1]))
    close(mzi(-1, 1j, 0, 0), [-1, 1j])
    assert mzi_matrix(0, 0).dtype == np.complex128


def test_swap_sign_and_zero():
    close(mzi(2+1j, -3j, np.pi/2, 0), [3j, 2+1j])
    close(mzi(0, 0, .3, .9), [0, 0])


@pytest.mark.parametrize('phi,expected', [(0, [0, 1]), (np.pi, [-1, 0])])
def test_interference(phi, expected):
    close(mzi(1/np.sqrt(2), 1/np.sqrt(2), np.pi/4, phi), expected)


def test_random_unitarity_power_inverse_and_linearity():
    rng = np.random.default_rng(42)
    for _ in range(100):
        theta, phi = rng.uniform(-2*np.pi, 2*np.pi, 2)
        T = mzi_matrix(theta, phi)
        x, y = rng.normal(size=(2,2)) + 1j*rng.normal(size=(2,2))
        out = np.asarray(mzi(*x, theta, phi))
        close(T.conj().T @ T, np.eye(2))
        close(np.vdot(out, out), np.vdot(x, x))
        close(T.conj().T @ out, x)
        close(mzi(*(x+2j*y), theta, phi), out+2j*np.asarray(mzi(*y, theta, phi)))


def test_embedding_and_input_unchanged():
    E = np.array([1, 2j, -3, 4-2j])
    original = E.copy()
    out = apply_mzi(E, 1, np.pi/2, 0)
    close(out, [1, 3, 2j, 4-2j])
    close(E, original)
    assert not np.shares_memory(out, E)


@pytest.mark.parametrize('bad', [np.nan, np.inf, 1j, [1], True])
def test_invalid_angles(bad):
    with pytest.raises(ValueError): mzi_matrix(bad, 0)
    with pytest.raises(ValueError): mzi_matrix(0, bad)


@pytest.mark.parametrize('E,m', [([[1,2]],0), ([1],0), ([1,2],-1), ([1,2],1), ([1,2],.5), ([1,np.inf],0), ([1,2],True)])
def test_invalid_fields_and_channels(E, m):
    with pytest.raises(ValueError): apply_mzi(E, m, 0, 0)


def test_scalar_interface_rejects_vectors():
    with pytest.raises(ValueError): mzi([1], [2], 0, 0)
