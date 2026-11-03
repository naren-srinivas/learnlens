"""
learnlens.fourier
==================
Empirical Boolean Fourier analysis and an estimator for the "leap
complexity" of a target function, based on the Merged-Staircase Property
(MSP) of Abbe, Boix-Adsera & Misiakiewicz, "The merged-staircase property:
a necessary and nearly sufficient condition for SGD learning of sparse
Boolean functions", COLT 2023.

Background: for f: {-1,+1}^d -> {-1,+1} depending on a support S with
|S|=k, the Fourier expansion is f(x) = sum_{T subseteq [d]} f_hat(T) chi_T(x)
where chi_T(x) = prod_{i in T} x_i. The MSP theory shows SGD can learn f
efficiently, incrementally discovering one relevant variable at a time,
if and only if (roughly) there is an ordering of S such that every prefix
subset has non-negligible correlation with some function of the previous
prefix -- informally, no "leap" in Fourier support is too large. Pure
k-sparse parity is the worst case (leap = k, only the full-support Fourier
coefficient is non-zero); staircase-structured functions such as our
decision lists have leap close to 1.

We estimate this empirically and cheaply via Monte Carlo Fourier
coefficient estimation restricted to subsets of the known support
(tractable since k <= ~8, so 2^k <= 256 subsets).
"""
import itertools
import numpy as np


def estimate_fourier_coefficients(bayes_fn, d, support, n_samples=20000, rng=None):
    """Monte Carlo estimate of f_hat(T) = E_x[f(x) * chi_T(x)] for every
    subset T of `support` (not of all [d] -- intractable and unnecessary
    since by construction f depends only on `support`).
    Returns dict: frozenset(T) -> estimated coefficient.
    """
    rng = rng or np.random.default_rng()
    X = rng.choice([-1.0, 1.0], size=(n_samples, d))
    y = bayes_fn(X)
    coeffs = {}
    support = list(support)
    for r in range(len(support) + 1):
        for subset in itertools.combinations(support, r):
            if subset:
                chi = np.prod(X[:, list(subset)], axis=1)
            else:
                chi = np.ones(n_samples)
            coeffs[frozenset(subset)] = float(np.mean(y * chi))
    return coeffs


def empirical_leap_complexity(coeffs, support, threshold=0.05):
    """Boolean leap: minimize the largest number of NEW coordinates added
    by a significant monomial to the UNION of preceding monomials.

    This is Definition 1 of Abbe et al., COLT 2023, not a nested chain.
    Returns None when thresholding misses coordinates in the supplied support;
    those missing coefficients cannot legitimately be invented as endpoints.
    Monte Carlo thresholding can still miss small terms or retain noise.
    """
    support = frozenset(support)
    k = len(support)
    if k == 0:
        return 0

    candidates = {frozenset(t) for t, c in coeffs.items() if abs(c) > threshold}
    if any(not t <= support for t in candidates):
        raise ValueError("Fourier terms must be subsets of support")
    if frozenset().union(*candidates) != support:
        return None
    candidates = sorted(candidates, key=len)

    def feasible(L):
        discovered = frozenset()
        changed = True
        while changed:
            changed = False
            for c in candidates:
                if 0 < len(c - discovered) <= L:
                    discovered = discovered | c
                    changed = True
        return discovered == support

    lo, hi = 1, k
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def leap_complexity_for_problem(problem, n_samples=20000, threshold=0.05, rng=None):
    """ORACLE diagnostic, not an estimator from an ordinary labeled dataset.
    Queries bayes_fn on paired counterfactual inputs and uses support_size
    metadata. Exponential in recovered support size. Boolean problems only.
    """
    rng = rng or np.random.default_rng()
    d = problem.d
    # Recover the (approximate) relevant support via single-variable
    # influence: Inf_i = E_x[(f(x) - f(x with bit i flipped))^2] / 4
    n_probe = 4000
    X = rng.choice([-1.0, 1.0], size=(n_probe, d))
    y = problem.bayes_fn(X)
    influences = np.zeros(d)
    for i in range(d):
        Xf = X.copy()
        Xf[:, i] *= -1
        yf = problem.bayes_fn(Xf)
        influences[i] = np.mean((y - yf) ** 2) / 4.0
    k = problem.theory.get("support_size", None)
    if k is None:
        k = int(np.sum(influences > 0.05))
        k = max(k, 1)
    support = list(np.argsort(-influences)[:k])
    coeffs = estimate_fourier_coefficients(problem.bayes_fn, d, support,
                                            n_samples=n_samples, rng=rng)
    leap = empirical_leap_complexity(coeffs, support, threshold=threshold)
    return leap, support, coeffs
