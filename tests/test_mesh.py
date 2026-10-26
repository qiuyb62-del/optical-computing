import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.mesh import propagate_mesh, reconstruct_mesh, mesh_layers


def close(a, b):
    assert_allclose(a, b, atol=1e-12, rtol=1e-12)


def test_manual_noncommuting_cascade_and_output_phase():
    # 手算：先交换通道0/1，再交换1/2，最后通道0乘i。
    x = np.array([1, 2j, -3], dtype=complex)
    original = x.copy()
    ops = [(0, np.pi/2, 0), (1, np.pi/2, 0)]
    phases = np.array([np.pi/2, 0, 0])
    close(propagate_mesh(x, ops, phases), [2, 3, 1])
    close(reconstruct_mesh(ops, phases), [[0, -1j, 0], [0, 0, -1], [1, 0, 0]])
    close(x, original)
    close(phases, [np.pi/2, 0, 0])


def test_empty_mesh_keeps_full_phase_and_supports_one_channel():
    close(propagate_mesh([2j], [], [np.pi/2]), [-2])
    close(reconstruct_mesh([], [.3, -.7]), np.diag(np.exp(1j*np.array([.3, -.7]))))
    close(propagate_mesh([0, 0], [], [3, 4]), [0, 0])


def test_layers_commute_only_disjoint_units():
    ops = [(0,.2,.4), (2,.8,.1), (1,.3,.5), (0,.1,.9)]
    layers = mesh_layers(ops, [0]*4)
    assert [len(layer) for layer in layers] == [2, 1, 1]
    x = np.array([1, -2j, 3+1j, 4])
    close(propagate_mesh(x, ops, [0]*4), propagate_mesh(x, sum(layers, []), [0]*4))
    assert not np.shares_memory(x, propagate_mesh(x, ops, [0]*4))


@pytest.mark.parametrize('ops,phases', [
    ([], []), ([], [[0]]), ([], [1j]), ([], [np.nan]), ([], ['0']),
    ([(0,0)], [0,0]), ([(1,0,0)], [0,0]), ([(True,0,0)], [0,0]),
    ([(.5,0,0)], [0,0]), ([(0,np.inf,0)], [0,0]), (None, [0]),
])
def test_invalid_mesh(ops, phases):
    for function in [reconstruct_mesh, mesh_layers]:
        with pytest.raises(ValueError): function(ops, phases)
    with pytest.raises(ValueError): propagate_mesh([0,0], ops, phases)


@pytest.mark.parametrize('x', [[1], [[1,2]], [np.inf,0]])
def test_bad_field(x):
    with pytest.raises(ValueError): propagate_mesh(x, [], [0,0])
