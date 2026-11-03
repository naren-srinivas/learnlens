import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learnlens import k_sparse_parity, k_decision_list, margin_halfspace
from learnlens.models import NumpyMLP
from learnlens.fourier import (
    estimate_fourier_coefficients, empirical_leap_complexity, leap_complexity_for_problem,
)
from learnlens.ntk import empirical_ntk_gram, kernel_target_alignment, ntk_alignment_at_init
from learnlens.pacbayes import pacbayes_bound
from learnlens.predictor import build_feature_matrix, fit_ols, leave_one_out_cv, fit_full_model


def test_leap_complexity_equals_k_for_pure_parity():
    for k in [2, 3, 4]:
        prob = k_sparse_parity(d=14, k=k, seed_support=0)
        leap, support, coeffs = leap_complexity_for_problem(
            prob, n_samples=12000, rng=np.random.default_rng(0))
        assert leap == k, f"expected leap={k} for pure parity, got {leap}"


def test_leap_complexity_is_low_for_decision_list():
    prob = k_decision_list(d=14, k=4, seed_support=0)
    leap, support, coeffs = leap_complexity_for_problem(
        prob, n_samples=12000, rng=np.random.default_rng(0))
    assert leap <= 2, f"expected low leap for a staircase-structured decision list, got {leap}"


def test_fourier_coefficient_of_full_support_parity_is_near_one():
    prob = k_sparse_parity(d=10, k=3, seed_support=1)
    # recover support via influence, same approach used internally
    leap, support, coeffs = leap_complexity_for_problem(
        prob, n_samples=15000, rng=np.random.default_rng(0))
    full_coeff = coeffs[frozenset(support)]
    assert abs(abs(full_coeff) - 1.0) < 0.1, f"expected |f_hat(S)|~=1, got {full_coeff}"


def test_kta_near_zero_for_random_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 8))
    y_random = rng.choice([-1.0, 1.0], size=80)
    mlp = NumpyMLP(hidden=32, seed=0)
    mlp._init_params(8)
    K = empirical_ntk_gram(mlp, X)
    kta = kernel_target_alignment(K, y_random)
    assert abs(kta) < 0.15, f"expected near-zero KTA for random labels, got {kta}"


def test_kta_ranks_problems_in_theoretically_expected_order():
    rng = np.random.default_rng(0)
    mlp_factory = lambda: NumpyMLP(hidden=48, seed=0)

    prob_parity = k_sparse_parity(d=14, k=4, seed_support=0)
    prob_dlist = k_decision_list(d=14, k=4, seed_support=0)
    prob_margin = margin_halfspace(d=14, margin=0.5, seed_w=0)

    kta_parity = np.mean([ntk_alignment_at_init(mlp_factory, prob_parity, n_probe=120,
                                                  rng=np.random.default_rng(s)) for s in range(3)])
    kta_dlist = np.mean([ntk_alignment_at_init(mlp_factory, prob_dlist, n_probe=120,
                                                 rng=np.random.default_rng(s)) for s in range(3)])
    kta_margin = np.mean([ntk_alignment_at_init(mlp_factory, prob_margin, n_probe=120,
                                                  rng=np.random.default_rng(s)) for s in range(3)])

    assert kta_margin > kta_parity, "margin halfspace should have higher pre-training KTA than parity"
    assert kta_dlist > kta_parity, "decision list should have higher pre-training KTA than parity"


def test_pacbayes_bound_is_valid_probability_and_monotone_with_n():
    rng = np.random.default_rng(0)
    prob = margin_halfspace(d=10, margin=0.5, seed_w=0)
    bounds = []
    for n in [200, 3000]:
        X, y = prob.sample(n, rng=rng)
        mlp = NumpyMLP(hidden=32, lr=0.1, epochs=100, batch_size=32, seed=0)
        mlp.fit(X, y)
        b = pacbayes_bound(mlp, X, y, rng=rng, n_mc=8)
        assert 0.0 <= b["bound"] <= 1.0
        bounds.append(b["bound"])
    # more data should generally give a tighter (smaller or equal) bound
    assert bounds[1] <= bounds[0] + 0.05


def test_predictor_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    records = []
    true_beta = np.array([1.0, 0.8, -1.5, 0.3])
    for i in range(12):
        leap = rng.integers(1, 6)
        kta = rng.uniform(0, 0.3)
        d = rng.choice([10, 20, 30])
        logd = np.log(d)
        noise = rng.normal(scale=0.05)
        log_n_star = true_beta @ np.array([1.0, leap, kta, logd]) + noise
        records.append(dict(leap=leap, kta_init=kta, d=d, n_star=np.exp(log_n_star)))
    fit = fit_full_model(records)
    assert fit["in_sample_r2"] > 0.9, f"expected near-perfect in-sample fit on synthetic linear data, got {fit['in_sample_r2']}"

    loo_r2, pairs = leave_one_out_cv(records)
    assert loo_r2 > 0.5, f"expected reasonable LOO R^2 on clean synthetic linear data, got {loo_r2}"


if __name__ == "__main__":
    import inspect
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
    print(f"{passed}/{len(tests)} passed")
