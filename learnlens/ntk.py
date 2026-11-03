"""
learnlens.ntk
==============
Empirical Neural Tangent Kernel (NTK) and kernel-target alignment (KTA),
computed AT INITIALIZATION, before any training. This gives a cheap,
pre-training predictor of learnability: Jacot et al. (NeurIPS 2018) show
that in the infinite-width limit, a network's training dynamics are
governed by its (fixed) NTK; the alignment between that kernel and the
target labels (Cristianini et al., "On Kernel-Target Alignment", 2001)
is a classical predictor of how easily a kernel method fits a given
target function -- high alignment means the kernel's dominant
eigendirections already correlate with y.

We compute the *exact* (not random-feature-approximated) empirical NTK
of our 2-layer tanh NumpyMLP by explicit per-example parameter gradients,
which is tractable at the small scale used throughout this project
(hidden <= 64, probe set <= a few hundred points).
"""
import numpy as np


def _per_example_gradients(mlp, X):
    """Return J, shape (n, P): the Jacobian of the scalar network output
    f(x) = tanh(z2) with respect to every parameter, evaluated at the
    MLP's CURRENT weights (call this right after _init_params for an
    at-initialization NTK, or after training for an empirical NTK at
    convergence).
    """
    n, d = X.shape
    hidden = mlp.W1.shape[1]
    z1 = X @ mlp.W1 + mlp.b1          # (n, hidden)
    a1 = np.tanh(z1)                    # (n, hidden)
    z2 = a1 @ mlp.W2 + mlp.b2          # (n,)
    pred = np.tanh(z2)                  # (n,)
    dpred_dz2 = 1 - pred ** 2            # (n,)

    # grad wrt W2 (n, hidden): dpred/dW2[h] = dpred_dz2 * a1[:,h]
    gW2 = dpred_dz2[:, None] * a1                       # (n, hidden)
    gb2 = dpred_dz2                                        # (n,)

    # grad wrt z1 (n, hidden): dpred/dz1[h] = dpred_dz2 * W2[h] * (1-a1^2)
    dz1 = dpred_dz2[:, None] * mlp.W2[None, :] * (1 - a1 ** 2)  # (n, hidden)
    gb1 = dz1                                               # (n, hidden)
    # grad wrt W1[j,h] = dz1[:,h] * X[:,j]  -> (n, d*hidden)
    gW1 = dz1[:, None, :] * X[:, :, None]                # (n, d, hidden)
    gW1 = gW1.reshape(n, d * hidden)

    J = np.concatenate([gW1, gb1, gW2, gb2[:, None]], axis=1)
    return J, pred


def empirical_ntk_gram(mlp, X):
    """Empirical NTK Gram matrix K (n x n): K_ij = <grad f(x_i), grad f(x_j)>."""
    J, _ = _per_example_gradients(mlp, X)
    return J @ J.T


def kernel_target_alignment(K, y, centered=False):
    """Cristianini et al. 2001 kernel-target alignment:
        A(K, y) = <K, yy^T>_F / (n * ||K||_F)
    (since ||yy^T||_F = n for y in {-1,+1}^n). Ranges roughly in [-1, 1];
    higher means the kernel's structure already correlates with the labels.
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if K.shape != (n, n) or not n:
        raise ValueError("Expected nonempty square Gram matrix matching labels")
    if centered:
        K = K - K.mean(axis=0)[None, :] - K.mean(axis=1)[:, None] + K.mean()
        y = y - y.mean()
    numerator = float(y @ K @ y)
    denom = float(y @ y) * np.linalg.norm(K, ord="fro")
    if denom < 1e-12:
        return 0.0
    return numerator / denom


def ntk_alignment_at_init(mlp_factory, problem, n_probe=150, rng=None):
    """Construct a fresh (unfit) MLP, initialize its parameters, and
    compute the kernel-target alignment of its empirical NTK against a
    probe sample from `problem` -- entirely BEFORE any training step.
    This is the pre-training learnability predictor used by LearnLens.
    """
    rng = rng or np.random.default_rng()
    X, y = problem.sample(n_probe, rng=rng)
    mlp = mlp_factory()
    mlp._init_params(X.shape[1])  # initialize weights only, no training
    K = empirical_ntk_gram(mlp, X)
    return kernel_target_alignment(K, y)
