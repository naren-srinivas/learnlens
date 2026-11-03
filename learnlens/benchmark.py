"""Budget-controlled parity experiments with paired data and explicit censoring."""
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm
from .baselines import GF2Parity
from .models import NumpyMLP
from .ntk import empirical_ntk_gram, kernel_target_alignment


def wilson_interval(errors, n, confidence=.95):
    if n <= 0 or not 0 <= errors <= n or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence")
    z = norm.ppf((1 + confidence) / 2)
    p = errors / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    radius = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def transition_interval(ns, passed):
    """Grid-local transition; no imputation of unobserved crossing times."""
    if len(ns) != len(passed) or not ns or any(a >= b for a,b in zip(ns, ns[1:])):
        raise ValueError("Expected matching, strictly increasing nonempty grid")
    for i in range(len(ns)):
        if all(passed[i:]):
            return dict(status="left_censored" if i == 0 else "interval",
                        lower=0 if i == 0 else ns[i-1], upper=ns[i])
    return dict(status="right_censored", lower=ns[-1], upper=None)


def run_benchmark(dimensions=(12, 24), ks=(2, 3, 4), ns=(64, 256, 1024),
                  seeds=(0, 1, 2), updates=400, hidden=32, n_test=1000,
                  target_error=.15, output=None):
    """Freeze update count across n; epochs vary. LR=.1 and batch=32.

    All n use prefixes of a shared training draw. Target supports, data and
    weight initialization vary across seeds. A disjoint validation sample
    selects grid crossings; a separate untouched test set is reported only.
    Confidence intervals are pointwise, not simultaneous transition bounds.
    """
    if updates < 1 or n_test < 1 or hidden < 1:
        raise ValueError("Positive updates, n_test and hidden required")
    ns = sorted(set(int(n) for n in ns))
    if not ns or ns[0] < 1 or not 0 < target_error < .5:
        raise ValueError("Invalid sample grid or target error")
    records, transitions = [], []
    for d in dimensions:
        for k in ks:
            from .problems import k_sparse_parity
            for seed in seeds:
                problem = k_sparse_parity(d=d, k=k, seed_support=seed)
                rng = np.random.default_rng(seed + 10000)
                X, y = problem.sample(max(ns), rng=rng)
                Xv, yv = problem.sample(n_test, rng=rng)
                Xt, yt = problem.sample(n_test, rng=rng)
                init = NumpyMLP(hidden=hidden, seed=seed)
                init._init_params(d)
                probe = min(100, n_test)
                K = empirical_ntk_gram(init, Xv[:probe])
                kta = kernel_target_alignment(K, yv[:probe])
                centered_kta = kernel_target_alignment(K, yv[:probe], centered=True)
                for n in ns:
                    models = [GF2Parity(), NumpyMLP(hidden=hidden, lr=.1,
                        epochs=math.ceil(updates/math.ceil(n/32)), batch_size=32,
                        seed=seed, max_updates=updates)]
                    for model in models:
                        t0 = time.perf_counter()
                        model.fit(X[:n], y[:n])
                        elapsed = time.perf_counter()-t0
                        errors = int(np.sum(model.predict(Xt) != yt))
                        val_errors = int(np.sum(model.predict(Xv) != yv))
                        val_ci = wilson_interval(val_errors, n_test)
                        records.append(dict(d=int(d), k=int(k), seed=int(seed), n=n,
                            learner="GF2" if isinstance(model, GF2Parity) else "SGD",
                            err_test=errors/n_test, test_error_ci=wilson_interval(errors, n_test),
                            err_validation=val_errors/n_test, validation_error_ci=val_ci,
                            passes_validation=val_ci[1] <= target_error,
                            updates=getattr(model, "updates_", 0), seconds=elapsed,
                            unique_solution=getattr(model, "unique_solution_", None),
                            kta_init=kta, centered_kta_init=centered_kta))
                for learner in ["GF2", "SGD"]:
                    rows = [r for r in records if (r['d'],r['k'],r['seed'],r['learner']) == (d,k,seed,learner)]
                    transitions.append(dict(d=int(d), k=int(k), seed=int(seed), learner=learner,
                        **transition_interval(ns, [r['passes_validation'] for r in rows])))
                print(f"d={d} k={k} seed={seed}: complete", flush=True)
    config = dict(dimensions=list(dimensions), ks=list(ks), ns=ns, seeds=list(seeds),
                  updates=updates, hidden=hidden, n_test=n_test, target_error=target_error)
    source_files = ['benchmark.py', 'models.py', 'problems.py', 'baselines.py', 'ntk.py']
    source_hashes = {f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in source_files}
    payload = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        config=config, config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()),
        source_sha256=source_hashes, records=records, transitions=transitions,
        limitations=["Synthetic noiseless parity only; no novelty or hardness proof",
            "Pointwise 95% Wilson intervals; no simultaneous crossing guarantee",
            "Validation determines crossings; test data is descriptive only",
            "Fixed updates controls optimization steps, not identical wall-clock cost",
            "Small default seed count; increase for research conclusions"])
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload
