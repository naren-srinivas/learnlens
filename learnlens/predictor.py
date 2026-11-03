"""
learnlens.predictor
=====================
The central new contribution of this project: instead of only describing
phase transitions after running a full (n) sweep, predict the transition
sample size n* from features that are CHEAP and computable either before
training (KTA at init) or from the problem definition alone (leap
complexity, ambient dimension), via a small linear model fit across many
problem instances and validated with leave-one-out cross-validation
(appropriate given the necessarily small number of distinct problem
instances we can afford to sweep exhaustively).

Model:
    log(n*) ~ beta0 + beta1 * leap + beta2 * KTA_init + beta3 * log(d)

Fit via ordinary least squares (closed form, no external ML framework
needed). We report leave-one-out cross-validated R^2, which is a fair
measure given the small sample size (each fold trains on all-but-one
problem instance and predicts the held-out one).
"""
import numpy as np


def build_feature_matrix(records):
    """records: list of dicts each with keys 'leap', 'kta_init', 'd', 'n_star'.
    Returns (X, y_log) with X columns [1, leap, kta_init, log(d)].
    """
    rows = []
    y = []
    for r in records:
        rows.append([1.0, r["leap"], r["kta_init"], np.log(r["d"])])
        y.append(np.log(r["n_star"]))
    return np.array(rows), np.array(y)


def fit_ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict(X, beta):
    return X @ beta


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return 1 - ss_res / ss_tot


def leave_one_out_cv(records):
    """Returns (loo_r2, per_point_predictions) where per_point_predictions
    is a list of (true_log_n_star, predicted_log_n_star) pairs, each
    predicted by a model trained on all OTHER points.
    """
    X, y = build_feature_matrix(records)
    n = len(records)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta = fit_ols(X[mask], y[mask])
        preds[i] = X[i] @ beta
    loo_r2 = r_squared(y, preds)
    pairs = list(zip(y.tolist(), preds.tolist()))
    return loo_r2, pairs


def fit_full_model(records):
    """Fit on all data (for reporting coefficients / in-sample fit;
    NOT for claiming predictive accuracy -- use leave_one_out_cv for that)."""
    X, y = build_feature_matrix(records)
    beta = fit_ols(X, y)
    y_pred = X @ beta
    in_sample_r2 = r_squared(y, y_pred)
    return dict(beta=beta.tolist(), in_sample_r2=in_sample_r2,
                feature_names=["intercept", "leap", "kta_init", "log_d"])
