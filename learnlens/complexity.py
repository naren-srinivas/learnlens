"""Post-hoc exploratory diagnostics, not learnability guarantees.

LI uses test error, gap and random-label fitting correlation. The separately
fitted curve exponent is NOT an ingredient of LI. A learner fit is only a
lower approximation to the Rademacher supremum, not a certified capacity bound.
"""
import numpy as np


def zero_one_error(y_true, y_pred):
    return float(np.mean(y_true != y_pred))


def empirical_rademacher(fit_predict_fn, X, n_draws=20, rng=None):
    """Algorithm-dependent random-label fitting correlation proxy:
        R_hat = E_sigma [ (1/n) * sup_h  sum_i sigma_i * h(x_i) ]
    approximated by re-fitting the learner to random +-1 "labels" sigma
    and measuring the best achievable correlation with those random
    labels. High R_hat means the hypothesis class can fit noise easily
    (high capacity relative to n) -- a classical overfitting-risk proxy.
    """
    rng = rng or np.random.default_rng()
    n = X.shape[0]
    correlations = []
    for _ in range(n_draws):
        sigma = rng.choice([-1.0, 1.0], size=n)
        pred = fit_predict_fn(X, sigma, X)  # fit on (X, sigma), predict on X
        correlations.append(np.mean(sigma * pred))
    return float(np.mean(correlations))


def fit_learning_curve_exponent(ns, errs, floor=1e-3):
    """Fit err(n) ~ C * n^-alpha in log-log space via least squares.
    Returns (alpha_hat, C_hat, r_squared). alpha_hat is the empirical
    sample-complexity exponent: larger alpha = faster-improving / easier.
    """
    ns = np.asarray(ns, dtype=float)
    errs = np.clip(np.asarray(errs, dtype=float), floor, None)
    logn, loge = np.log(ns), np.log(errs)
    A = np.vstack([logn, np.ones_like(logn)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, loge, rcond=None)
    alpha_hat = -slope
    C_hat = np.exp(intercept)
    pred = A @ np.array([slope, intercept])
    ss_res = np.sum((loge - pred) ** 2)
    ss_tot = np.sum((loge - loge.mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot
    return float(alpha_hat), float(C_hat), float(r2)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def learnability_index(err_test, gen_gap, rademacher_hat, a=6.0, b=4.0, c=2.0):
    """Composite Learnability Index in [0, 1]. See module docstring."""
    margin_term = 1.0 - err_test / 0.5  # 1 at perfect, 0 at chance, negative if worse than chance
    score = a * margin_term - b * abs(gen_gap) - c * max(rademacher_hat, 0.0)
    return float(_sigmoid(score))


def find_phase_transition(ns, li_values, threshold=0.5):
    """Return the smallest n at which LI(n) first crosses `threshold`
    and stays >= threshold for all larger tested n (robust crossing).
    Returns None if no such crossing exists in the tested range.
    """
    ns = np.asarray(ns)
    li = np.asarray(li_values)
    if ns.ndim != 1 or len(ns) == 0 or len(ns) != len(li) or np.any(np.diff(ns) <= 0) or not np.isfinite(li).all():
        raise ValueError("Expected finite matching values and strictly increasing ns")
    for i in range(len(ns)):
        if li[i] >= threshold and np.all(li[i:] >= threshold):
            return float(ns[i])
    return None
