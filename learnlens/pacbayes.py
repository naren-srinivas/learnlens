"""Finite-grid PAC-Bayes bound for a Gaussian Gibbs classifier.

The zero-mean prior variance grid MUST be fixed before inspecting data.
A union bound accounts for selecting the prior variance and Hoeffding
accounts for finite Monte Carlo estimation of the empirical Gibbs risk.
This does not certify the deterministic trained network. Non-vacuous means
below 1; beating balanced chance means below .5.
"""
import numpy as np


def _param_vector_norm_sq(mlp):
    return (np.sum(mlp.W1 ** 2) + np.sum(mlp.b1 ** 2)
            + np.sum(mlp.W2 ** 2) + float(mlp.b2) ** 2)


def _num_params(mlp):
    return mlp.W1.size + mlp.b1.size + mlp.W2.size + 1


def _mc_perturbed_error(mlp, X, y, sigma, n_mc, rng):
    errs = []
    for _ in range(n_mc):
        W1p = mlp.W1 + rng.normal(scale=sigma, size=mlp.W1.shape)
        b1p = mlp.b1 + rng.normal(scale=sigma, size=mlp.b1.shape)
        W2p = mlp.W2 + rng.normal(scale=sigma, size=mlp.W2.shape)
        b2p = mlp.b2 + rng.normal(scale=sigma)
        z1 = X @ W1p + b1p
        a1 = np.tanh(z1)
        z2 = a1 @ W2p + b2p
        pred = np.where(z2 > 0, 1.0, -1.0)
        errs.append(np.mean(pred != y))
    return float(np.mean(errs))


def pacbayes_bound(mlp, X, y, sigmas=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3),
                    delta=0.05, n_mc=15, rng=None):
    """Search over `sigmas`, return the tightest (smallest) resulting
    PAC-Bayes bound, along with diagnostic fields.
    """
    sigmas = tuple(sigmas)
    if not sigmas or any(not np.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError("sigmas must be a nonempty positive finite grid")
    if not 0 < delta < 1 or not isinstance(n_mc, int) or n_mc < 1:
        raise ValueError("Expected 0 < delta < 1 and positive integer n_mc")
    if len(X) < 2 or len(X) != len(y):
        raise ValueError("Expected matching X,y with at least two observations")
    rng = rng or np.random.default_rng()
    n = X.shape[0]
    theta_norm_sq = _param_vector_norm_sq(mlp)
    P = _num_params(mlp)

    best = None
    # Grid of data-independent priors must be fixed before observing data.
    # Half the failure budget is for PAC-Bayes, half for MC estimation.
    delta_each = delta / (2 * len(sigmas))
    mc_radius = float(np.sqrt(np.log(1 / delta_each) / (2 * n_mc)))
    for sigma in sigmas:
        kl = theta_norm_sq / (2 * sigma ** 2)
        mc_err = _mc_perturbed_error(mlp, X, y, sigma, n_mc, rng)
        complexity_term = np.sqrt(max(kl + np.log(2 * np.sqrt(n) / delta_each), 0) / (2 * n))
        empirical_upper = min(mc_err + mc_radius, 1.0)
        bound = min(empirical_upper + complexity_term, 1.0)
        rec = dict(sigma=sigma, kl=float(kl), mc_error=mc_err,
                   mc_radius=mc_radius, empirical_risk_upper=empirical_upper,
                   complexity_term=float(complexity_term), bound=float(bound))
        if best is None or bound < best["bound"]:
            best = rec

    best["num_params"] = P
    best["n"] = n
    best["non_vacuous"] = best["bound"] < 1.0
    best["beats_chance"] = best["bound"] < 0.5
    best["delta"] = delta
    best["n_mc"] = n_mc
    best["prior_grid_size"] = len(sigmas)
    best["certified_classifier"] = "Gaussian Gibbs; not the deterministic network"
    return best
