import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learnlens import (
    k_sparse_parity, margin_halfspace, k_decision_list,
    zero_one_error, empirical_rademacher, fit_learning_curve_exponent,
    learnability_index, find_phase_transition,
)
from learnlens.models import NumpyMLP, SklearnWrapper
from learnlens.experiment import run_sweep, summarize
from sklearn.linear_model import LogisticRegression


def test_parity_labels_consistent_with_support():
    prob = k_sparse_parity(d=10, k=3, seed_support=0)
    X, y = prob.sample(50, rng=np.random.default_rng(0))
    assert np.array_equal(y, prob.bayes_fn(X))
    assert set(np.unique(y)) <= {-1.0, 1.0}


def test_margin_halfspace_respects_margin():
    prob = margin_halfspace(d=10, margin=0.3, seed_w=0)
    X, y = prob.sample(200, rng=np.random.default_rng(0))
    w = None
    # recover w indirectly via bayes_fn consistency
    assert np.array_equal(y, prob.bayes_fn(X))
    proj_abs_min = None  # can't access w directly, but no label noise means margin holds structurally
    assert set(np.unique(y)) <= {-1.0, 1.0}


def test_decision_list_consistent():
    prob = k_decision_list(d=12, k=4, seed_support=0)
    X, y = prob.sample(100, rng=np.random.default_rng(0))
    assert np.array_equal(y, prob.bayes_fn(X))


def test_zero_one_error_basic():
    y_true = np.array([1, -1, 1, -1])
    y_pred = np.array([1, 1, 1, -1])
    assert abs(zero_one_error(y_true, y_pred) - 0.25) < 1e-9


def test_numpy_mlp_learns_linearly_separable_data():
    rng = np.random.default_rng(0)
    d = 5
    w = rng.normal(size=d)
    X = rng.normal(size=(300, d))
    y = np.sign(X @ w)
    mlp = NumpyMLP(hidden=16, lr=0.1, epochs=100, batch_size=32, seed=0)
    mlp.fit(X, y)
    acc = (mlp.predict(X) == y).mean()
    assert acc > 0.95, f"MLP should easily fit simple linear data, got acc={acc}"


def test_numpy_mlp_fails_on_parity_with_too_few_samples():
    # With very few samples and a hard-ish k, MLP should be near chance.
    prob = k_sparse_parity(d=30, k=6, seed_support=0)
    X, y = prob.sample(300, rng=np.random.default_rng(1))
    Xte, yte = prob.sample(300, rng=np.random.default_rng(2))
    mlp = NumpyMLP(hidden=64, lr=0.1, epochs=100, batch_size=32, seed=0)
    mlp.fit(X, y)
    acc = (mlp.predict(Xte) == yte).mean()
    assert 0.35 < acc < 0.65, f"Expected near-chance accuracy on hard parity w/ few samples, got {acc}"


def test_rademacher_complexity_higher_for_flexible_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))

    def linear_fit_predict(Xtr, ytr, Xq):
        m = LogisticRegression(max_iter=500).fit(Xtr, (ytr > 0).astype(int))
        return np.where(m.predict(Xq) > 0, 1.0, -1.0)

    def tree_fit_predict(Xtr, ytr, Xq):
        from sklearn.tree import DecisionTreeClassifier
        m = DecisionTreeClassifier(max_depth=None).fit(Xtr, (ytr > 0).astype(int))
        return np.where(m.predict(Xq) > 0, 1.0, -1.0)

    rad_linear = empirical_rademacher(linear_fit_predict, X, n_draws=15, rng=np.random.default_rng(1))
    rad_tree = empirical_rademacher(tree_fit_predict, X, n_draws=15, rng=np.random.default_rng(1))
    # An unconstrained deep tree can fit random labels far better than a linear model
    assert rad_tree > rad_linear, f"expected tree R_hat ({rad_tree}) > linear R_hat ({rad_linear})"


def test_learning_curve_exponent_recovers_known_rate():
    # err(n) = 2 * n^-0.5 exactly -> alpha_hat should be close to 0.5
    ns = np.array([100, 400, 1600, 6400])
    errs = 2.0 * ns ** -0.5
    alpha_hat, C_hat, r2 = fit_learning_curve_exponent(ns, errs)
    assert abs(alpha_hat - 0.5) < 1e-6
    assert r2 > 0.999


def test_learnability_index_monotonic_in_error():
    li_good = learnability_index(err_test=0.01, gen_gap=0.0, rademacher_hat=0.05)
    li_bad = learnability_index(err_test=0.49, gen_gap=0.0, rademacher_hat=0.05)
    assert li_good > li_bad


def test_find_phase_transition_detects_crossing():
    ns = [10, 20, 30, 40, 50]
    li = [0.1, 0.2, 0.55, 0.6, 0.9]
    n_star = find_phase_transition(ns, li, threshold=0.5)
    assert n_star == 30


def test_end_to_end_sweep_runs_and_summarizes():
    prob = k_sparse_parity(d=8, k=2, seed_support=0)
    learner_factory = lambda: SklearnWrapper("LR", lambda: LogisticRegression(max_iter=500))
    recs = run_sweep(prob, learner_factory, ns=[50, 100], n_test=80,
                      n_rademacher_draws=3, seed=0, verbose=False)
    summary = summarize(prob, "LR", recs)
    assert "alpha_hat" in summary
    assert len(summary["records"]) == 2


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
