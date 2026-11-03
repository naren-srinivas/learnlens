"""
learnlens.experiment
=====================
Runs the (problem x learner x n) sweep and computes, for every cell:
  - test error, generalization gap
  - empirical Rademacher complexity
  - Learnability Index LI(n)
Then fits the learning-curve exponent alpha_hat and locates the
empirical phase-transition sample size n_star, comparing it against
the theoretical sample complexity recorded on the Problem object.
"""
import numpy as np
from .complexity import (
    zero_one_error, empirical_rademacher, fit_learning_curve_exponent,
    learnability_index, find_phase_transition,
)


def _fit_predict_generic(learner_factory):
    """Wrap a learner factory into the (X, y, X_query) -> pred signature
    needed by empirical_rademacher (fresh model each call)."""
    def fit_predict_fn(X, y, X_query):
        model = learner_factory()
        model.fit(X, y)
        return model.predict(X_query)
    return fit_predict_fn


def run_sweep(problem, learner_factory, ns, n_test=400, noise=0.0,
              n_rademacher_draws=10, seed=0, verbose=True):
    """learner_factory: zero-arg callable returning a fresh, unfit learner
    (so every n gets an independently trained model)."""
    rng = np.random.default_rng(seed)
    X_test, y_test = problem.sample(n_test, noise=0.0, rng=rng)  # clean test set

    records = []
    for n in ns:
        X_train, y_train = problem.sample(n, noise=noise, rng=rng)
        model = learner_factory()
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        err_train = zero_one_error(y_train, train_pred)
        err_test = zero_one_error(y_test, test_pred)
        gen_gap = err_test - err_train

        rad = empirical_rademacher(
            _fit_predict_generic(learner_factory), X_train,
            n_draws=n_rademacher_draws, rng=rng,
        )
        li = learnability_index(err_test, gen_gap, rad)

        rec = dict(n=n, err_train=err_train, err_test=err_test,
                   gen_gap=gen_gap, rademacher=rad, learnability_index=li)
        records.append(rec)
        if verbose:
            print(f"  n={n:6d}  err_test={err_test:.3f}  gap={gen_gap:+.3f}  "
                  f"R_hat={rad:+.3f}  LI={li:.3f}")
    return records


def summarize(problem, learner_name, records):
    ns = [r["n"] for r in records]
    errs = [r["err_test"] for r in records]
    alpha_hat, C_hat, r2 = fit_learning_curve_exponent(ns, errs)
    li_values = [r["learnability_index"] for r in records]
    n_star = find_phase_transition(ns, li_values)
    return dict(
        problem=problem.name,
        learner=learner_name,
        theory=problem.theory,
        alpha_hat=alpha_hat,
        curve_fit_r2=r2,
        empirical_n_star=n_star,
        final_test_error=errs[-1],
        records=records,
    )
