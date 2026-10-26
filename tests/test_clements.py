import itertools
import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.clements import decompose_clements
from src.mesh import propagate_mesh, reconstruct_mesh, mesh_layers


def random_unitary(n, rng):
    z = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))[None, :]


def verify(U):
    before = U.copy()
    n = len(U)
    ops, phases = decompose_clements(U)
    actual = reconstruct_mesh(ops, phases)
    assert np.linalg.norm(actual-U, 'fro')/np.linalg.norm(U, 'fro') <= 1e-10
    assert np.linalg.norm(actual.conj().T@actual-np.eye(n), 'fro') <= 1e-10
    assert_allclose(U, before, atol=0, rtol=0)
    assert len(ops) == n*(n-1)//2
    assert phases.shape == (n,)
    assert np.all((phases >= 0) & (phases < 2*np.pi))
    assert all(0 <= op[2] < 2*np.pi for op in ops)
    layers = mesh_layers(ops, phases)
    assert len(layers) <= n
    for layer in layers:
        used = [m for op in layer for m in (op[0], op[0]+1)]
        assert len(used) == len(set(used))
    # 实际逐器件传播所有基向量，验证全部列；比较复数，保留全局相位。
    for x in np.eye(n, dtype=complex):
        assert_allclose(propagate_mesh(x, ops, phases), U@x, atol=1e-10, rtol=0)
    rng = np.random.default_rng(123)
    x = rng.normal(size=n)+1j*rng.normal(size=n)
    y = propagate_mesh(x, ops, phases)
    assert np.linalg.norm(y-U@x) <= 1e-10*max(1,np.linalg.norm(U@x))
    assert_allclose(np.vdot(y,y), np.vdot(x,x), atol=1e-10, rtol=1e-12)
    assert_allclose(propagate_mesh(np.zeros(n), ops, phases), 0, atol=0, rtol=0)
    assert_allclose(reconstruct_mesh(sum(layers, []), phases), actual, atol=1e-12, rtol=0)


@pytest.mark.parametrize('n', [1,2,3,4,5,8,16])
def test_random_complex_unitaries(n):
    rng = np.random.default_rng(42)
    for _ in range(20): verify(random_unitary(n, rng))


@pytest.mark.parametrize('n', [1,2,3,4,5,8])
def test_identity_diagonal_permutation_fourier(n):
    verify(np.eye(n, dtype=complex))
    verify(np.diag(np.exp(1j*np.linspace(-np.pi, 2*np.pi, n))))
    verify(np.eye(n, dtype=complex)[::-1])
    k = np.arange(n)
    verify(np.exp(2j*np.pi*np.outer(k,k)/n)/np.sqrt(n))
    verify(np.exp(.731j)*np.eye(n))


def test_all_four_channel_permutations():
    for permutation in itertools.permutations(range(4)):
        verify(np.eye(4, dtype=complex)[list(permutation)])


def test_tiny_nonzero_and_near_swap():
    # 独立构造含极小元素的实旋转，避免零分母，也不丢失符号。
    for angle in [1e-14, -1e-14, np.pi/2-1e-14, np.pi/2+1e-14]:
        c,s = np.cos(angle),np.sin(angle)
        U = np.eye(4,dtype=complex)
        U[:2,:2] = [[c,-s],[s,c]]
        verify(U)


def test_known_four_channel_rectangular_layout():
    ops, phases = decompose_clements(np.eye(4))
    assert [op[0] for op in ops] == [0,2,1,0,2,1]
    assert [[op[0] for op in layer] for layer in mesh_layers(ops, phases)] == [[0,2],[1],[0,2],[1]]


@pytest.mark.parametrize('U', [[], [1,0], [[1,0]], np.zeros((2,2)),
    np.diag([1,2]), [[np.nan]], [[np.inf]], [[1e300]], [['bad']]])
def test_rejects_invalid_unitaries(U):
    with pytest.raises(ValueError): decompose_clements(U)
