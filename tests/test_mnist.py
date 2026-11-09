import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
import pytest
from src.mnist_data import PixelPCA, parse_idx, training_split, load_mnist
from src.batch_optics import trace_batch, optical_forward_batch
from src.ai_inference import ClassifierWeights, digital_forward, optical_forward
from src.svd_matmul import compile_svd, trace_svd
from src.nonideal import NonidealConfig, realize_nonideal, trace_nonideal

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size", [2, 4, 16])
@pytest.mark.parametrize("nonideal", [False, True])
def test_batch_is_scalar_complex_propagation_with_frozen_device(size, nonideal):
    rng = np.random.default_rng(331)
    W = rng.normal(size=(size, size)) + 1j*rng.normal(size=(size, size))
    X = rng.normal(size=(7, size)) + 1j*rng.normal(size=(7, size))
    X[0] = 0
    original = X.copy()
    circuit = compile_svd(W)
    trace = trace_svd
    if nonideal:
        circuit = realize_nonideal(circuit, NonidealConfig(phase_std_rad=.1, loss_db_per_mzi=.1,
                                                          phase_bits=6, coupler_through=.49), seed=992)
        trace = trace_nonideal
    snapshot = circuit.to_dict()
    batch = trace_batch(X, circuit)
    singles = [trace(x, circuit) for x in X]
    for key in ("after_input_mesh", "after_attenuation", "optical_output", "restored_output"):
        np.testing.assert_allclose(getattr(batch, key), np.stack([getattr(x, key) for x in singles]), atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(trace_batch(X[::-1], circuit).restored_output, batch.restored_output[::-1], atol=1e-12)
    np.testing.assert_array_equal(X, original)
    assert snapshot == circuit.to_dict()
    if not nonideal:
        np.testing.assert_allclose(batch.restored_output, X @ W.T, atol=2e-12, rtol=1e-12)


def test_ten_class_pipeline_orientation_readout_and_no_recompilation():
    rng = np.random.default_rng(22)
    weights = ClassifierWeights(rng.normal(size=(16,16)), rng.normal(size=16),
                                 rng.normal(size=(10,16)), rng.normal(size=10))
    X = rng.normal(size=(13,16))
    circuit = compile_svd(weights.W1)
    digital = digital_forward(X, weights)
    batch = optical_forward_batch(X, weights, circuit)
    np.testing.assert_allclose(batch["logits"], digital["logits"], atol=2e-11)
    assert batch["passivity_passed"]
    device = realize_nonideal(circuit, NonidealConfig(phase_std_rad=.1), seed=7)
    noisy = optical_forward_batch(X, weights, device)
    serial = optical_forward(X, weights, device)
    np.testing.assert_allclose(noisy["logits"], serial["logits"], atol=2e-11)
    np.testing.assert_allclose(noisy["powers"], serial["powers"], atol=2e-11)
    changed = ClassifierWeights(np.eye(16)*200, weights.b1, weights.W2, weights.b2)
    np.testing.assert_array_equal(noisy["linear"], optical_forward_batch(X, changed, device)["linear"])
    assert np.linalg.norm(noisy["linear"].imag) > .1


@pytest.mark.parametrize("bad", [[], [[1, 2, 3]], [[1, np.nan]], [[1, np.inf]], [1, 2]])
def test_batch_rejects_invalid_fields(bad):
    with pytest.raises(ValueError):
        trace_batch(bad, compile_svd(np.eye(2)))


def test_idx_parser_shapes_labels_and_corruption(tmp_path):
    raw = struct.pack(">IIII", 2051, 2, 28, 28) + bytes(range(256))*6 + bytes(32)
    parsed = parse_idx(raw, count=2, images=True)
    assert parsed.shape == (2,28,28) and parsed.dtype == np.uint8
    label_raw = struct.pack(">II", 2049, 2) + bytes([3,9])
    np.testing.assert_array_equal(parse_idx(label_raw, count=2, images=False), [3,9])
    for invalid in (raw[:-1], raw+b"x", b"x", struct.pack(">IIII", 1,2,28,28)+raw[16:]):
        with pytest.raises(ValueError): parse_idx(invalid, count=2, images=True)
    with pytest.raises(ValueError):
        parse_idx(struct.pack(">II", 2049, 2)+bytes([3,10]), count=2, images=False)
    with pytest.raises(FileNotFoundError): load_mnist(tmp_path)
    (tmp_path / "train-images-idx3-ubyte.gz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"): load_mnist(tmp_path, download=True)


def test_training_validation_partition_and_fixed_class_counts():
    y = np.repeat(np.arange(10), 6000)
    a, b = training_split(y), training_split(y)
    assert [len(a[k]) for k in ("train", "validation")] == [50000, 10000]
    np.testing.assert_array_equal(np.sort(np.concatenate(list(a.values()))), np.arange(60000))
    np.testing.assert_array_equal(np.bincount(y[a["validation"]]), np.full(10,1000))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["train"], training_split(y, 34)["train"])


def test_pca_matches_independent_svd_and_does_not_learn_from_transform(tmp_path):
    rng = np.random.default_rng(912)
    train = rng.integers(0, 256, size=(45,28,28), dtype=np.uint8)
    test = rng.integers(0, 256, size=(5,28,28), dtype=np.uint8)
    pca = PixelPCA.fit(train)
    X = train.reshape(45,784).astype(float)/255
    np.testing.assert_allclose(pca.mean, X.mean(axis=0))
    _, s, vh = np.linalg.svd(X-X.mean(axis=0), full_matrices=False)
    np.testing.assert_allclose(pca.components.T@pca.components, vh[:16].T@vh[:16], atol=2e-12)
    np.testing.assert_allclose(pca.explained_variance_ratio, s[:16]**2/np.sum(s**2), atol=1e-12)
    before = pca.mean.copy()
    transformed = pca.transform(test, batch_size=2)
    np.testing.assert_allclose(transformed, (test.reshape(5,784)/255-pca.mean)@pca.components.T, atol=1e-12)
    np.testing.assert_allclose(pca.transform(train).mean(0), 0, atol=1e-12)
    np.testing.assert_array_equal(before, pca.mean)
    pca.save(tmp_path / "pca.npz")
    np.testing.assert_array_equal(PixelPCA.load(tmp_path/"pca.npz").transform(test), pca.transform(test))
    with pytest.raises(ValueError): PixelPCA.fit(np.zeros((20,28,28),dtype=np.uint8))
    with pytest.raises(ValueError): pca.transform(test.astype(float))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional training dependency PyTorch not installed")
def test_mnist_minibatch_training_reproducible_and_restores_torch_state():
    import torch
    from src.mnist_training import train_mnist, MNISTTrainingConfig
    rng = np.random.default_rng(872)
    X, y = rng.normal(size=(120,16)), np.tile(np.arange(10),12)
    before_rng = torch.random.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    config = MNISTTrainingConfig(max_epochs=3, patience=2, batch_size=23)
    a, ta = train_mnist(X[:100], y[:100], X[100:], y[100:], seed=3, config=config)
    b, tb = train_mnist(X[:100], y[:100], X[100:], y[100:], seed=3, config=config)
    assert a == b and ta == tb
    assert ta["torch_numpy_max_absolute_error"] < 1e-12
    assert ta["best_epoch"] == min(ta["history"], key=lambda r:r["validation_loss"])["epoch"]
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert before_threads == torch.get_num_threads()


def test_offline_mnist_replay_needs_no_torch_matplotlib_or_raw_dataset(tmp_path):
    if not (ROOT/"progress/06_mnist/models/seed_20261109.json").exists():
        pytest.skip("saved MNIST model is not generated yet")
    script = """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('torch','matplotlib','torchvision'):
        raise ImportError('blocked optional dependency')
    return original(name,*args,**kwargs)
builtins.__import__ = blocked
from demo_week06 import replay
r = replay()
assert r['versions']['ideal_optical']['prediction_agreement'] == 1
assert len(r['test_indices']) == 20
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)


def test_saved_full_test_predictions_reproduce_every_metric():
    import csv
    folder = ROOT/"progress/06_mnist"
    if not (folder/"results.json").exists():
        pytest.skip("saved MNIST experiment is not generated yet")
    report = json.loads((folder/"results.json").read_text())
    assert report["passed"]
    for name, checksum in report["artifacts_sha256"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == checksum
    with np.load(folder/"predictions.npz",allow_pickle=False) as pred:
        y = pred["test_labels"]
        assert len(y) == 10000
        np.testing.assert_array_equal(pred["test_indices"], np.arange(10000))
        for run in report["runs"]:
            d, i = pred[f"digital_{run['seed']}"], pred[f"ideal_{run['seed']}"]
            np.testing.assert_array_equal(d, i)
            assert np.mean(d == y) == run["digital"]["accuracy"]
        rows = list(csv.DictReader((folder/"devices.csv").open()))
        assert len(rows) == report["counts"]["nonideal_configurations"]
        for row in rows:
            p = pred[row["prediction_key"]]
            assert np.mean(p == y) == float(row["accuracy"])
            confusion = np.zeros((10,10),dtype=int)
            np.add.at(confusion,(y,p),1)
            np.testing.assert_array_equal(confusion,json.loads(row["confusion_matrix"]))
