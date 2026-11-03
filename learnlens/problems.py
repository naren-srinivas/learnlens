"""
learnlens.problems
===================
Synthetic hypothesis classes drawn from computational learning theory,
chosen because they have KNOWN theoretical sample-complexity / hardness
characterizations, which lets us validate empirical estimates against
ground truth.

Every generator returns X in {-1,+1}^d (or R^d) and y in {-1,+1}, together
with a `bayes_fn` describing the ground-truth labeling rule and a
`theory` dict of known theoretical facts (used later to sanity-check
our empirical estimators).
"""
import numpy as np


class Problem:
    """A synthetic learning problem: samples from a fixed distribution
    over (x, y) with a known ground-truth rule."""

    def __init__(self, name, d, sampler, bayes_fn, theory):
        self.name = name
        self.d = d
        self.sampler = sampler
        self.bayes_fn = bayes_fn
        self.theory = theory

    def sample(self, n, noise=0.0, rng=None):
        if not isinstance(n, (int, np.integer)) or n < 1 or not 0 <= noise <= 1:
            raise ValueError("Expected positive integer n and noise in [0,1]")
        rng = rng or np.random.default_rng()
        X, y = self.sampler(n, rng)
        if noise > 0:
            flip = rng.random(n) < noise
            y = np.where(flip, -y, y)
        return X, y


def k_sparse_parity(d=40, k=3, seed_support=None):
    """k-sparse parity over {-1,+1}^d: y = product of x_i for i in S, |S|=k.

    THEORY: PAC-learnable with O(d + log(1/delta)) noiseless samples via Gaussian
    elimination (statistically easy), but believed to require samples
    or time scaling like d^Omega(k) for correlational-statistical-query
    (CSQ) algorithms such as SGD on typical architectures unless the
    network is initialized/scaled to exploit higher-order correlations
    (Barak et al. 2022). This is THE canonical statistical-computational
    gap example in learning theory.
    """
    if not isinstance(d, int) or not isinstance(k, int) or not 1 <= k <= d:
        raise ValueError("Expected integer 1 <= k <= d")
    rng_support = np.random.default_rng(seed_support if seed_support is not None else 0)
    support = rng_support.choice(d, size=k, replace=False)

    def sampler(n, rng):
        X = rng.choice([-1.0, 1.0], size=(n, d))
        y = np.prod(X[:, support], axis=1)
        return X, y

    def bayes_fn(X):
        return np.prod(X[:, support], axis=1)

    theory = {
        "statistical_sample_complexity": f"O(d) via Gaussian elimination "
                                          f"over F_2 (d={d}), independent of k",
        "csq_hardness": "CSQ lower bounds depend on tolerance and query budget. SGD rates require architecture, initialization and optimization assumptions; a failed finite run is not a hardness proof.",
        "hypothesis_class": "Unknown k-sparse parity; target support hidden from learners",
        "support_size": k,
        "ambient_dim": d,
    }
    return Problem(f"{k}-sparse-parity(d={d})", d, sampler, bayes_fn, theory)


def margin_halfspace(d=20, margin=0.5, seed_w=None):
    """Linearly separable data with a fixed margin gamma.

    THEORY: sample complexity for halfspace/SVM learning scales as
    O(1/gamma^2) independent of ambient dimension d (margin bound,
    Bartlett & Mendelson 2002 Rademacher-complexity style bound),
    i.e. EASY and margin-limited, not dimension-limited.
    """
    if not isinstance(d, int) or d < 1 or not np.isfinite(margin) or not 0 < margin <= 3:
        raise ValueError("Expected positive integer d and 0 < margin <= 3")
    rng_w = np.random.default_rng(seed_w if seed_w is not None else 1)
    w = rng_w.normal(size=d)
    w = w / np.linalg.norm(w)

    def sampler(n, rng):
        X = rng.normal(size=(n, d))
        proj = X @ w
        # Reject-resample points inside the margin band to guarantee margin
        mask = np.abs(proj) < margin
        while mask.any():
            X[mask] = rng.normal(size=(mask.sum(), d))
            proj = X @ w
            mask = np.abs(proj) < margin
        y = np.sign(proj)
        return X, y

    def bayes_fn(X):
        return np.sign(X @ w)

    theory = {
        "statistical_sample_complexity": "Truncated Gaussian with unbounded norm: a dimension-free bounded-radius margin bound does not directly apply. Error and confidence parameters are omitted from informal rates.",
        "csq_hardness": "none — gradient-based linear/kernel methods achieve "
                         "the statistical rate (no computational-statistical gap)",
        "vc_dimension": d + 1,
        "margin": margin,
        "ambient_dim": d,
    }
    return Problem(f"margin-halfspace(d={d},gamma={margin})", d, sampler, bayes_fn, theory)


def k_decision_list(d=30, k=4, seed_support=None):
    """A depth-k decision list over k of d boolean features
    (nested if/else on literals). PAC-learnable in poly(d) time & samples
    by a specific poly-time algorithm (Rivest 1987) AND poly-time by SGD
    on suitable architectures — included as an "easy for both" control
    condition, contrasting with parity's CSQ-hardness.
    """
    if not isinstance(d, int) or not isinstance(k, int) or not 1 <= k <= d:
        raise ValueError("Expected integer 1 <= k <= d")
    rng_support = np.random.default_rng(seed_support if seed_support is not None else 2)
    support = rng_support.choice(d, size=k, replace=False)
    signs = rng_support.choice([-1.0, 1.0], size=k)

    def rule(x_row):
        for idx, s in zip(support, signs):
            if x_row[idx] == s:
                return s
        return -signs[-1]

    def sampler(n, rng):
        X = rng.choice([-1.0, 1.0], size=(n, d))
        y = np.array([rule(row) for row in X])
        return X, y

    def bayes_fn(X):
        return np.array([rule(row) for row in X])

    theory = {
        "statistical_sample_complexity": f"O(k log d) = O({k}*log({d})) "
                                          f"(Occam/VC bound for decision lists of length k)",
        "csq_hardness": "low — decision lists are efficiently CSQ-learnable "
                         "(Kearns 1998); SGD is expected to track the statistical rate",
        "hypothesis_class": "Length-k decision lists over d unknown coordinates; complexity scales with k log d",
        "support_size": k,
        "ambient_dim": d,
    }
    return Problem(f"{k}-decision-list(d={d})", d, sampler, bayes_fn, theory)


PROBLEM_REGISTRY = {
    "parity": k_sparse_parity,
    "margin_halfspace": margin_halfspace,
    "decision_list": k_decision_list,
}
