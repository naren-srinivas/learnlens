"""
learnlens.models
=================
Learners under test. Includes a from-scratch NumPy MLP (mini-batch SGD,
manual backprop) so we have a genuine "deep learning" representative
without any external DL framework dependency.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class SklearnWrapper:
    """Uniform interface: fit(X,y) -> self ; predict(X) -> {-1,+1}."""
    def __init__(self, name, estimator_fn):
        self.name = name
        self.estimator_fn = estimator_fn
        self.model = None

    def fit(self, X, y):
        self.model = self.estimator_fn()
        self.model.fit(X, (y > 0).astype(int))
        return self

    def predict(self, X):
        pred = self.model.predict(X)
        return np.where(pred > 0, 1.0, -1.0)


class NumpyMLP:
    """2-layer MLP, tanh hidden units, trained with mini-batch SGD and
    manual backprop -- a minimal, dependency-free 'deep learning' learner.
    Deliberately kept small/standard-init to reproduce the well-known
    failure mode on CSQ-hard targets like parity (Barak et al. 2022).
    """
    def __init__(self, hidden=64, lr=0.05, epochs=300, batch_size=32, seed=0, max_updates=None):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.max_updates = max_updates
        self.name = f"NumpyMLP(h={hidden})"

    def _init_params(self, d):
        rng = np.random.default_rng(self.seed)
        scale1 = np.sqrt(2.0 / d)
        scale2 = np.sqrt(2.0 / self.hidden)
        self.W1 = rng.normal(scale=scale1, size=(d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(scale=scale2, size=(self.hidden,))
        self.b2 = 0.0

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, z2

    def fit(self, X, y):
        n, d = X.shape
        self._init_params(d)
        self.updates_ = 0
        rng = np.random.default_rng(self.seed + 1)
        for epoch in range(self.epochs):
            idx = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                if self.max_updates is not None and self.updates_ >= self.max_updates:
                    return self
                b_idx = idx[start:start + self.batch_size]
                Xb, yb = X[b_idx], y[b_idx]
                z1, a1, z2 = self._forward(Xb)
                pred = np.tanh(z2)
                # squared-hinge-ish smooth loss: d(loss)/d(pred)
                grad_pred = (pred - yb) * (1 - pred ** 2)  # tanh output + MSE
                grad_W2 = a1.T @ grad_pred / len(b_idx)
                grad_b2 = grad_pred.mean()
                grad_a1 = np.outer(grad_pred, self.W2)
                grad_z1 = grad_a1 * (1 - a1 ** 2)
                grad_W1 = Xb.T @ grad_z1 / len(b_idx)
                grad_b1 = grad_z1.mean(axis=0)

                self.W2 -= self.lr * grad_W2
                self.b2 -= self.lr * grad_b2
                self.W1 -= self.lr * grad_W1
                self.b1 -= self.lr * grad_b1
                self.updates_ += 1
        return self

    def predict(self, X):
        _, _, z2 = self._forward(X)
        return np.where(z2 > 0, 1.0, -1.0)

    def predict_score(self, X):
        _, _, z2 = self._forward(X)
        return z2


def make_learners():
    """Return the standard battery of learners used in every experiment."""
    return [
        SklearnWrapper("LogisticRegression", lambda: LogisticRegression(max_iter=2000)),
        SklearnWrapper("KernelSVM(rbf)", lambda: SVC(kernel="rbf", C=1.0, gamma="scale")),
        SklearnWrapper("DecisionTree(d<=6)", lambda: DecisionTreeClassifier(max_depth=6)),
        NumpyMLP(hidden=64, lr=0.05, epochs=200, batch_size=32),
    ]
