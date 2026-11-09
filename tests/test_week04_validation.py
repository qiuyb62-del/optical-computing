import json
import numpy as np
import pytest
from verify_week04 import run_experiments, save_csv, save_svg, stats


def test_small_experiment_reproducible_and_distinguishes_experimental_error(tmp_path):
    a=run_experiments(sizes=(2,),samples=2,trials=2,seed=8)
    b=run_experiments(sizes=(2,),samples=2,trials=2,seed=8)
    assert a==b
    assert a['passed']
    assert len(a['conditions'])==16
    baseline=a['conditions'][0]
    assert baseline['matrix_relative_error']['max']<1e-10
    loss=next(r for r in a['conditions'] if r['condition']=='loss_0.5')
    assert loss['fraction_exceeding_ideal_vector_tolerance']==1
    assert loss['validation_passed']
    np.testing.assert_allclose(loss['matrix_relative_error']['median'],1-10**(-1/20),atol=1e-13)
    json.dumps(a,allow_nan=False)
    save_csv(a,tmp_path/'summary.csv')
    save_svg(a,tmp_path/'curves.svg')
    assert len((tmp_path/'summary.csv').read_text().splitlines())==17
    assert 'No hardware measurements' in (tmp_path/'curves.svg').read_text()


@pytest.mark.parametrize('kwargs',[{'samples':0},{'trials':0},{'seed':-1},{'sizes':()},
                                  {'sizes':(0,)},{'sizes':(2,2)}])
def test_invalid_experiment_settings(kwargs):
    with pytest.raises(ValueError): run_experiments(**kwargs)


def test_stats_are_empirical_quantiles():
    s=stats([0,1,2,3,4])
    assert s['median']==2 and s['count']==5
    assert s['p05']==pytest.approx(.2)
    assert s['p95']==pytest.approx(3.8)
    with pytest.raises(ValueError): stats([float('nan')])
