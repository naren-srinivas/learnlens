import numpy as np
import pytest
from learnlens.baselines import GF2Parity
from learnlens.fourier import empirical_leap_complexity
from learnlens.models import NumpyMLP
from learnlens.ntk import _per_example_gradients, kernel_target_alignment
from learnlens.pacbayes import pacbayes_bound
from learnlens.benchmark import transition_interval, wilson_interval, run_benchmark
from learnlens import k_sparse_parity


def test_gf2_recovers_hidden_parity_and_generalizes():
    problem = k_sparse_parity(d=24, k=7, seed_support=91)
    X,y = problem.sample(100, rng=np.random.default_rng(4))
    Xt,yt = problem.sample(500, rng=np.random.default_rng(5))
    model = GF2Parity().fit(X,y)
    assert model.unique_solution_
    assert model.coef_.sum() == 7
    assert np.array_equal(model.predict(Xt), yt)


def test_gf2_exposes_rank_deficiency_and_rejects_inconsistent_labels():
    X = np.ones((3,4))
    model = GF2Parity().fit(X,np.ones(3))
    assert not model.unique_solution_
    with pytest.raises(ValueError, match='Inconsistent'):
        GF2Parity().fit(X,np.array([1.,-1.,1.]))


def test_leap_uses_union_not_nested_chain():
    terms = {frozenset([0]):1., frozenset([0,1]):1., frozenset([1,2]):1.}
    assert empirical_leap_complexity(terms,[0,1,2]) == 1
    assert empirical_leap_complexity({frozenset([0,1,2]):1.},[0,1,2]) == 3
    assert empirical_leap_complexity({frozenset([0]):1.},[0,1,2]) is None


def test_ntk_manual_jacobian_matches_finite_differences():
    model = NumpyMLP(hidden=3,seed=5)
    model._init_params(2)
    X = np.array([[.2,-.5],[1.,.1]])
    J,_ = _per_example_gradients(model,X)
    columns = []
    for name in ['W1','b1','W2','b2']:
        value = np.asarray(getattr(model,name)).copy()
        for index in np.ndindex(value.shape):
            plus,minus = value.copy(),value.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            setattr(model,name,plus)
            fp = np.tanh(model.predict_score(X))
            setattr(model,name,minus)
            fm = np.tanh(model.predict_score(X))
            columns.append((fp-fm)/(2e-6))
        setattr(model,name,float(value) if value.ndim == 0 else value)
    assert np.allclose(J,np.array(columns).T,atol=1e-7)


def test_centered_kta_removes_constant_label_signal():
    assert kernel_target_alignment(np.ones((5,5)),np.ones(5)) == 1.
    assert kernel_target_alignment(np.ones((5,5)),np.ones(5),centered=True) == 0.


def test_updates_are_fixed_across_sample_sizes():
    for n in [50,300]:
        X,y = k_sparse_parity(d=6,k=2).sample(n,rng=np.random.default_rng(1))
        model = NumpyMLP(hidden=4,epochs=50,max_updates=20).fit(X,y)
        assert model.updates_ == 20


def test_pacbayes_accounts_for_mc_and_prior_selection():
    X,y = k_sparse_parity(d=4,k=2).sample(50,rng=np.random.default_rng(1))
    model = NumpyMLP(hidden=4,epochs=3).fit(X,y)
    result = pacbayes_bound(model,X,y,sigmas=(.3,),n_mc=5,rng=np.random.default_rng(7))
    assert result['bound'] >= min(1.,result['mc_error']+result['complexity_term'])
    assert result['mc_radius'] > 0
    assert result['non_vacuous'] == (result['bound'] < 1)
    with pytest.raises(ValueError):
        pacbayes_bound(model,X,y,sigmas=())


def test_censoring_and_wilson():
    assert transition_interval([10,20],[False,False]) == dict(status='right_censored',lower=20,upper=None)
    assert transition_interval([10,20],[True,True])['status'] == 'left_censored'
    assert transition_interval([10,20,30],[True,False,True]) == dict(status='interval',lower=20,upper=30)
    assert 0 <= wilson_interval(0,100)[0] < wilson_interval(0,100)[1] < .05


def test_benchmark_artifact_has_provenance_and_no_imputed_transitions(tmp_path):
    path = tmp_path/'benchmark.json'
    data = run_benchmark(dimensions=(6,),ks=(2,),ns=(16,32),seeds=(0,),updates=5,hidden=4,n_test=100,output=path)
    assert path.exists() and len(data['records']) == 4
    assert len(data['config_sha256']) == 64
    assert all(r['updates'] == 5 for r in data['records'] if r['learner'] == 'SGD')
    assert all(r['upper'] is None for r in data['transitions'] if r['status'] == 'right_censored')
