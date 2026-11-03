"""Structure-aware noiseless parity baseline; no hidden-support access."""
import numpy as np


class GF2Parity:
    """Gaussian elimination over F2, setting free coefficients to zero."""
    name = "GF2 parity solver"

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or not len(X):
            raise ValueError("Expected nonempty X (n,d) and y (n,)")
        if not np.isin(X, [-1, 1]).all() or not np.isin(y, [-1, 1]).all():
            raise ValueError("GF2Parity requires +/-1 inputs and labels")
        A = np.column_stack((X < 0, y < 0)).astype(np.uint8)
        n, d = X.shape
        pivots, rank = [], 0
        for col in range(d):
            available = np.flatnonzero(A[rank:, col])
            if not len(available):
                continue
            pivot = rank + available[0]
            A[[rank, pivot]] = A[[pivot, rank]]
            rows = np.flatnonzero(A[:, col])
            rows = rows[rows != rank]
            A[rows] ^= A[rank]
            pivots.append(col)
            rank += 1
            if rank == n:
                break
        if np.any((~A[:, :d].any(axis=1)) & (A[:, d] == 1)):
            raise ValueError("Inconsistent parity equations; labels may contain noise")
        self.coef_ = np.zeros(d, dtype=np.uint8)
        self.coef_[pivots] = A[:rank, d]
        self.rank_ = rank
        self.unique_solution_ = rank == d
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(self.coef_) or not np.isin(X, [-1, 1]).all():
            raise ValueError("Expected +/-1 inputs with training dimension")
        bits = (X < 0).astype(np.int64) @ self.coef_ % 2
        return np.where(bits, -1.0, 1.0)
