import json
import numpy as np
import pytest
from src.ai_data import load_iris, stratified_split, Standardizer, DATA_PATH
from src.ai_inference import (ClassifierWeights, digital_forward, optical_forward,
                              classification_metrics, compare_forward)
from src.svd_matmul import compile_svd
from src.nonideal import NonidealConfig, realize_nonideal


def example_weights():
    # Deliberately nonsymmetric: a transpose error cannot pass this fixture.
    return ClassifierWeights([[1,2,-3,4],[0,-2,1,3],[2,1,0,-1],[3,-1,4,2]],
                             [-.2,.3,1,-.7], [[1,-2,3,0],[0,1,2,-1],[-1,3,0,2]], [.3,-.4,.5])


def test_pinned_dataset_shape_corrected_rows_and_corruption(tmp_path):
    X, y = load_iris()
    assert X.shape == (150, 4)
    np.testing.assert_array_equal(np.bincount(y), [50,50,50])
    np.testing.assert_array_equal(X[34], [4.9,3.1,1.5,.2])
    np.testing.assert_array_equal(X[37], [4.9,3.6,1.4,.1])
    corrupt = tmp_path / "bad.data"
    corrupt.write_bytes(DATA_PATH.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_iris(corrupt)


def test_split_partition_stratification_and_reproducibility():
    X, y = load_iris()
    a, b = stratified_split(y, 12), stratified_split(y, 12)
    np.testing.assert_array_equal(np.sort(np.concatenate(list(a.values()))), np.arange(150))
    for key, expected in (("train",30),("validation",10),("test",10)):
        np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(np.bincount(y[a[key]]), [expected]*3)
    assert not np.array_equal(a["test"], stratified_split(y, 13)["test"])


def test_standardization_uses_only_training_and_preserves_input():
    X, y = load_iris()
    split = stratified_split(y, 4)
    original = X.copy()
    s = Standardizer.fit(X[split["train"]])
    changed = X.copy()
    changed[split["test"]] += 1e6
    assert s == Standardizer.fit(changed[split["train"]])
    standardized = s.transform(X[split["train"]])
    np.testing.assert_allclose(standardized.mean(0), 0, atol=1e-14)
    np.testing.assert_allclose(standardized.std(0), 1, atol=1e-14)
    np.testing.assert_array_equal(X, original)
    constant = Standardizer.fit(np.ones((2,4)))
    np.testing.assert_array_equal(constant.transform(np.ones((1,4))), np.zeros((1,4)))


def test_ideal_optical_orientation_bias_relu_head_and_complex_phase():
    weights = example_weights()
    X = np.array([[1,-2,3,.5],[-3,2,0,1],[0,0,0,0]])
    reference = digital_forward(X, weights)
    ideal = compile_svd(weights.W1)
    out = optical_forward(X, weights, ideal)
    np.testing.assert_allclose(out["linear"], X @ np.asarray(weights.W1).T, atol=1e-12)
    np.testing.assert_allclose(out["hidden"], reference["hidden"], atol=1e-12)
    np.testing.assert_allclose(out["logits"], reference["logits"], atol=1e-11)
    np.testing.assert_allclose(out["raw_optical"] * ideal.recovery_scale, out["linear"])
    assert out["passivity_passed"]
    assert compare_forward(out, reference)["prediction_agreement"] == 1
    # Changing reference W1 cannot alter propagation through an already compiled device.
    other = ClassifierWeights(np.eye(4)*100, weights.b1, weights.W2, weights.b2)
    np.testing.assert_array_equal(optical_forward(X, other, ideal)["linear"], out["linear"])


def test_fixed_nonideal_device_reused_for_all_inputs_and_real_readout():
    weights = example_weights()
    X = np.array([[1,-2,3,.5],[-3,2,0,1]])
    device = realize_nonideal(compile_svd(weights.W1), NonidealConfig(phase_std_rad=.1, loss_db_per_mzi=.2), seed=31)
    snapshot = device.to_dict()
    out = optical_forward(X, weights, device)
    singles = np.concatenate([optical_forward(x[None,:], weights, device)["linear"] for x in X])
    np.testing.assert_array_equal(out["linear"], singles)
    np.testing.assert_array_equal(out["linear"][::-1], optical_forward(X[::-1], weights, device)["linear"])
    assert device.to_dict() == snapshot
    assert np.linalg.norm(out["linear"].imag) > .01
    np.testing.assert_allclose(out["hidden"], np.maximum(out["linear"].real + weights.b1, 0))
    assert out["passivity_passed"]


def test_confusion_axes_accuracy_and_stable_cross_entropy():
    logits = np.array([[1000,0,0],[0,1000,0],[0,1000,0],[0,0,1000]])
    m = classification_metrics(np.array([0,2,1,2]), logits)
    assert m["accuracy"] == .75
    assert m["confusion_matrix"] == [[1,0,0],[0,1,0],[0,1,1]]
    assert m["cross_entropy"] == 250
    assert compare_forward({"linear": np.zeros((2,4)), "hidden": np.zeros((2,4)), "logits": np.zeros((2,3))},
                           {"linear": np.zeros((2,4)), "hidden": np.zeros((2,4)), "logits": np.zeros((2,3))})["linear_relative_error"] is None


@pytest.mark.parametrize("bad", [np.ones((4,)), np.ones((2,3)), [[1,2,3,np.nan]], np.ones((2,4), complex)])
def test_bad_features_rejected(bad):
    with pytest.raises(ValueError): digital_forward(bad, example_weights())


def test_weights_are_immutable_and_json_roundtrip():
    W = np.eye(4)
    w = ClassifierWeights(W, np.zeros(4), np.zeros((3,4)), np.zeros(3))
    W[0,0]=100
    assert w.W1[0][0] == 1
    assert ClassifierWeights(**json.loads(json.dumps(w.to_dict()))) == w
    with pytest.raises(ValueError): ClassifierWeights(np.ones((3,4)), w.b1, w.W2, w.b2)


def test_training_reproducible_best_validation_checkpoint_and_rng_restored():
    torch = pytest.importorskip("torch")
    from src.ai_training import train_classifier, TrainingConfig
    X, y = load_iris()
    split = stratified_split(y, 7)
    s = Standardizer.fit(X[split["train"]])
    args = (s.transform(X[split["train"]]), y[split["train"]], s.transform(X[split["validation"]]), y[split["validation"]])
    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    w, run = train_classifier(*args, seed=7, config=TrainingConfig(max_epochs=40, patience=5))
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.get_num_threads() == threads
    w2, run2 = train_classifier(*args, seed=7, config=TrainingConfig(max_epochs=40, patience=5))
    assert w == w2 and run == run2
    val_loss = classification_metrics(args[3], digital_forward(args[2], w)["logits"])["cross_entropy"]
    assert val_loss == pytest.approx(run["best_validation_loss"], abs=1e-14)
    assert run["best_validation_loss"] <= min(p["validation_loss"] for p in run["history"]) + 1e-8
    assert run["torch_numpy_max_absolute_error"] < 1e-12


def test_torch_weight_orientation_matches_optical_independently():
    torch = pytest.importorskip("torch")
    w = example_weights()
    X = np.array([[1,-2,3,.5],[-3,2,0,1]])
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    expected = torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(t(X), t(w.W1), t(w.b1))), t(w.W2), t(w.b2))
    np.testing.assert_allclose(optical_forward(X,w,compile_svd(w.W1))["logits"],expected.numpy(),atol=1e-11)


def test_small_experiment_saved_replay_and_pairing(tmp_path):
    pytest.importorskip("torch")
    from verify_week05 import run_experiments, save_csv
    from src.ai_training import TrainingConfig
    from demo_week05 import replay
    report, records, checkpoints = run_experiments(seeds=(7,), trials=2, config=TrainingConfig(max_epochs=3), output_dir=tmp_path)
    assert report["passed"]
    assert len(records) == 18
    assert report["counts"]["nonideal_test_predictions"] == 540
    assert len({r["device_seed"] for r in records if r["trial"] == 0}) == 1
    deterministic = [r for r in records if r["condition"] == "bits_4"]
    assert deterministic[0]["predictions"] == deterministic[1]["predictions"]
    assert deterministic[0]["linear_relative_error"] == deterministic[1]["linear_relative_error"]
    example = replay(checkpoints[0])
    saved = json.loads((tmp_path/"models/seed_7.json").read_text())
    assert example == replay(saved)
    expected = next(r for r in records if r["condition"] == "combined" and r["trial"] == 0)
    assert example["versions"]["nonideal_combined"]["predictions"] == expected["predictions"]
    json.dumps(report, allow_nan=False)
    save_csv(records, tmp_path/"devices.csv")
    assert len((tmp_path/"devices.csv").read_text().splitlines()) == 19


@pytest.mark.parametrize("kwargs", [{"seeds": ()}, {"seeds": (1,1)}, {"trials": 0}, {"seeds": (-1,)}])
def test_bad_experiment_args(kwargs):
    from verify_week05 import run_experiments
    with pytest.raises(ValueError): run_experiments(**kwargs)
