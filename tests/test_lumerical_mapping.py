import numpy as np
import pytest

from lumerical.experiments import _case_result
from lumerical.interconnect import (
    Port,
    _transmission,
    absolute_error,
    extract_forward,
    relative_error,
    resolve_lumerical_paths,
)
from lumerical.time_domain import set_encoded_vector
from lumerical.validation import (
    compare_complex,
    fde_phase_metrics,
    phase_and_loss_changes,
    power_ratio_mesh_converged,
    recent_phase_converged,
    strict_passivity,
    time_domain_passed,
)


class FakeApp:
    def __init__(self):
        self.values = {}

    def setnamed(self, name, prop, value):
        self.values[(name, prop)] = value


def test_encoded_vector_normalizes_and_keeps_signs_and_quadrature():
    app = FakeApp()
    controls = [(f"att{i}", f"ph{i}") for i in range(4)]
    x = np.array([2.0, -1.0, 0.5j, 0.0], dtype=complex)
    g = set_encoded_vector(app, controls, x)
    assert g == 2.0
    np.testing.assert_allclose(app.values[("att0", "attenuation")], 0.0, atol=1e-12)
    np.testing.assert_allclose(app.values[("att1", "attenuation")], -20 * np.log10(0.5))
    np.testing.assert_allclose(app.values[("att2", "attenuation")], -20 * np.log10(0.25))
    assert app.values[("att3", "attenuation")] == 300.0
    assert app.values[("ph0", "phase shift")] == pytest.approx(0.0)
    assert app.values[("ph1", "phase shift")] == pytest.approx(np.pi)
    assert app.values[("ph2", "phase shift")] == pytest.approx(np.pi / 2)
    assert app.values[("ph3", "phase shift")] == 0.0


def test_complex_relative_and_absolute_errors_do_not_align_phase():
    target = np.array([1.0 + 1j, -0.5j])
    assert relative_error(target, target) == 0.0
    assert absolute_error(target, target) == 0.0
    shifted = target * np.exp(1j * 0.2)
    assert relative_error(shifted, target) > 0.1
    assert absolute_error(shifted, target) > 0.0


def test_case_result_uses_absolute_criterion_for_zero_target():
    zero = _case_result("zero", np.zeros((2, 2), complex), np.zeros((2, 2), complex))
    assert zero["criterion"] == "absolute"
    assert zero["passed"] is True
    assert zero["relative_error"] is None
    assert "optical_target" not in zero
    assert "relative_error_to_optical_target" not in zero

    bad = _case_result("bad", np.ones((2, 2), complex), np.zeros((2, 2), complex))
    assert bad["criterion"] == "absolute"
    assert bad["passed"] is False

    good = _case_result("good", np.eye(2, dtype=complex), np.eye(2, dtype=complex))
    assert good["criterion"] == "relative"
    assert good["passed"] is True


def test_compare_complex_near_zero_uses_absolute_floor_and_rejects_nan():
    near_zero = compare_complex(np.array([1e-11]), np.zeros(1), scale=1.0)
    assert near_zero["criterion"] == "absolute"
    assert near_zero["passed"] is True
    bad_zero = compare_complex(np.array([1e-8]), np.zeros(1), scale=1.0)
    assert bad_zero["passed"] is False
    nan = compare_complex(np.array([np.nan]), np.array([1.0]))
    assert nan["passed"] is False
    assert nan["criterion"] == "invalid"


def test_time_domain_overall_requires_frequency_and_restored_wx():
    good = {"optical_Hx": {"passed": True}, "restored_Wx": {"passed": True}}
    bad_scale = {"optical_Hx": {"passed": True}, "restored_Wx": {"passed": False}}
    assert time_domain_passed(True, [good]) is True
    assert time_domain_passed(False, [good]) is False
    assert time_domain_passed(True, []) is False
    assert time_domain_passed(True, [good, bad_scale]) is False


def test_transmission_never_converts_missing_or_empty_to_zero():
    class Missing:
        def getresult(self, name, result):
            raise RuntimeError("missing")

    class NoField:
        def getresult(self, name, result):
            return {"frequency": [1.0]}

    class Empty:
        def getresult(self, name, result):
            return {"TE transmission": []}

    class NonFinite:
        def getresult(self, name, result):
            return {"TE transmission": [np.nan]}

    for app in (Missing(), NoField(), Empty(), NonFinite()):
        with pytest.raises(RuntimeError):
            _transmission(app, "ONA_1", 1)


class FakeForwardApp:
    def __init__(self, mode):
        self.mode = mode

    def switchtodesign(self):
        return None

    def connect(self, *args):
        return None

    def disconnect(self, *args):
        return None

    def run(self):
        return None

    def getresult(self, name, result):
        if self.mode == "missing":
            raise RuntimeError("Can not find result 'transmission' in provider")
        if self.mode == "api_error":
            raise RuntimeError("solver failed")
        if self.mode == "no_field":
            return {"frequency": [1.0]}
        if self.mode == "empty":
            return {"TE transmission": []}
        if self.mode == "nan":
            return {"TE transmission": [np.nan]}
        raise AssertionError(self.mode)


def test_extract_forward_only_falls_back_for_missing_result():
    inputs = [Port("in0", "p"), Port("in1", "p")]
    outputs = [Port("out0", "p"), Port("out1", "p")]
    diagnostics = {}
    H = extract_forward(FakeForwardApp("missing"), "ONA", inputs, outputs,
                        zero_power_map=np.full((2, 2), 1e-13),
                        diagnostics=diagnostics)
    np.testing.assert_array_equal(H, np.zeros((2, 2), dtype=complex))
    assert diagnostics["complex_fully_measured"] is False
    assert len(diagnostics["fallbacks"]) == 4
    assert diagnostics["amplitude_upper_bound"] == pytest.approx(np.sqrt(1e-13))


def test_extract_forward_rejects_invalid_data_even_with_low_power():
    inputs = [Port("in0", "p"), Port("in1", "p")]
    outputs = [Port("out0", "p"), Port("out1", "p")]
    small = np.full((2, 2), 1e-13)
    for mode in ("api_error", "no_field", "empty", "nan"):
        with pytest.raises(RuntimeError):
            extract_forward(FakeForwardApp(mode), "ONA", inputs, outputs,
                            zero_power_map=small)


def test_extract_forward_rejects_bad_power_map():
    inputs = [Port("in0", "p"), Port("in1", "p")]
    outputs = [Port("out0", "p"), Port("out1", "p")]
    for bad in (np.full((1, 1), 1e-13), np.full((2, 2), -1.0),
                np.full((2, 2), np.nan)):
        with pytest.raises(ValueError):
            extract_forward(FakeForwardApp("missing"), "ONA", inputs, outputs,
                            zero_power_map=bad)


def test_extract_forward_rejects_missing_without_zero_proof():
    inputs = [Port("in0", "p"), Port("in1", "p")]
    outputs = [Port("out0", "p"), Port("out1", "p")]
    with pytest.raises(RuntimeError):
        extract_forward(FakeForwardApp("missing"), "ONA", inputs, outputs)
    with pytest.raises(RuntimeError):
        extract_forward(FakeForwardApp("missing"), "ONA", inputs, outputs,
                        zero_power_map=np.full((2, 2), 1e-6))


def test_mnist_preflight_fails_before_starting_lumerical(tmp_path, monkeypatch):
    import lumerical.run_mnist16 as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.preflight(20261109, download=False)


def test_strict_passivity_is_pointwise_without_clipping():
    assert strict_passivity([0.0, 1.0])["passed"] is True
    for bad in ([-1e-12], [1.0 + 5e-7], [1.0000049707], [np.nan], [np.inf]):
        assert strict_passivity(bad)["passed"] is False


def test_power_ratio_mesh_requires_three_levels_and_all_pairs():
    two = {"2": {"T_net_ratio": [0.99]}, "3": {"T_net_ratio": [0.9901]}}
    assert power_ratio_mesh_converged(two, ["2", "3"])["passed"] is False
    three = {
        "2": {"T_net_ratio": [0.99]},
        "3": {"T_net_ratio": [0.9901]},
        "4": {"T_net_ratio": [0.9902]},
    }
    assert power_ratio_mesh_converged(three, ["2", "3", "4"])["passed"] is True
    three["4"]["T_net_ratio"] = [0.995]
    assert power_ratio_mesh_converged(three, ["2", "3", "4"])["passed"] is False


def test_recent_phase_convergence_ignores_only_early_coarse_pair():
    checks = [
        {"from_step_nm": 20.0, "to_step_nm": 10.0, "phase_converged": False},
        {"from_step_nm": 10.0, "to_step_nm": 7.5, "phase_converged": True},
        {"from_step_nm": 7.5, "to_step_nm": 5.0, "phase_converged": True},
    ]
    assert recent_phase_converged(checks, run_count=4)["passed"] is True
    assert recent_phase_converged(checks[:2], run_count=3)["passed"] is False
    assert recent_phase_converged(checks[:1], run_count=2)["passed"] is False
    checks[-1]["phase_converged"] = False
    assert recent_phase_converged(checks, run_count=4)["passed"] is False


def test_fde_phase_metric_rejects_large_neff_change():
    bad = fde_phase_metrics(2.4458481078, 2.4481005875,
                            length=100e-6, wavelength=1.55e-6, phase_limit=0.01)
    assert bad["delta_neff"] > 2e-3
    assert bad["phase_delta_rad"] > 0.9
    assert bad["phase_converged"] is False
    good = fde_phase_metrics(2.445848, 2.445848 + 1e-6,
                             length=100e-6, wavelength=1.55e-6, phase_limit=0.01)
    assert good["phase_converged"] is True


def test_fde_phase_uses_real_part_and_reports_loss_separately():
    same_real = phase_and_loss_changes(2.4458 + 0.001j, 2.4458 + 0.002j,
                                       length=100e-6, wavelength=1.55e-6)
    assert same_real["phase_delta_rad"] == 0.0
    assert same_real["power_loss_delta_db"] > 0.0
    bad = phase_and_loss_changes(2.4458, float("nan"),
                                 length=100e-6, wavelength=1.55e-6)
    assert bad["finite"] is False
    assert bad["phase_converged"] is False


def test_resolve_lumerical_paths_supports_env_api(tmp_path, monkeypatch):
    api = tmp_path / "install" / "api" / "python"
    api.mkdir(parents=True)
    (api / "lumapi.py").write_text("# fake API\n", encoding="utf-8")
    monkeypatch.setenv("LUMERICAL_API", str(api))
    install, resolved_api, python = resolve_lumerical_paths()
    assert install == tmp_path / "install"
    assert resolved_api == api
    assert python == tmp_path / "install" / "python-3.13.1" / "python.exe"
