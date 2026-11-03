"""LearnLens research workbench. Run: streamlit run app.py"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from learnlens import k_sparse_parity, margin_halfspace, k_decision_list
from learnlens.models import NumpyMLP, SklearnWrapper
from learnlens.baselines import GF2Parity
from learnlens.benchmark import wilson_interval
from learnlens.complexity import empirical_rademacher, learnability_index

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="LearnLens | Research workbench", page_icon="🔬", layout="wide")
st.title("LearnLens")
st.markdown("### When does more data help — and when does the learner matter?")
st.write("A reproducible CPU workbench for synthetic learning theory. Compare sample size, optimization budget and inductive bias using a from-scratch NumPy network and an exact parity solver.")
st.caption("Exploratory research software · No new theorem or established novelty claim · Legacy paper and figures require revision")

with st.sidebar:
    st.header("Live experiment")
    problem_type = st.selectbox("Problem", ["k-sparse parity", "margin halfspace", "k-decision list"])
    d = st.slider("Ambient dimension d", 4, 40, 16)
    k = None
    margin = .5
    if problem_type != "margin halfspace":
        k = st.slider("Sparsity / list length k", 1, min(8, d), min(3, d))
    else:
        margin = st.slider("Margin", .05, 1., .5)
    n = st.select_slider("Training samples n", [50, 100, 250, 500, 1000, 2000, 5000], value=500)
    options = ["LogisticRegression", "KernelSVM(rbf)", "DecisionTree", "NumpyMLP"]
    if problem_type == "k-sparse parity":
        options.append("GF2 exact parity")
    learner_name = st.selectbox("Learner", options)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=0)
    updates = st.slider("SGD update budget", 100, 2000, 400, step=100)
    run = st.button("Run experiment", type="primary")

live, evidence, methods = st.tabs(["Live laboratory", "Audited benchmark", "Methods & limitations"])
with live:
    st.subheader("Train, compare, export")
    st.write("Choose a configuration and run an experiment. The error interval describes this model on an independent test draw; it does not capture variability between training runs.")
    if run:
        if problem_type == "k-sparse parity":
            problem = k_sparse_parity(d=d, k=k, seed_support=int(seed))
        elif problem_type == "k-decision list":
            problem = k_decision_list(d=d, k=k, seed_support=int(seed))
        else:
            problem = margin_halfspace(d=d, margin=margin, seed_w=int(seed))
        def factory():
            if learner_name == "GF2 exact parity":
                return GF2Parity()
            if learner_name == "NumpyMLP":
                import math
                return NumpyMLP(hidden=32, lr=.1, epochs=math.ceil(updates/math.ceil(n/32)), seed=int(seed), max_updates=updates)
            if learner_name == "LogisticRegression":
                return SklearnWrapper("LR", lambda: LogisticRegression(max_iter=2000))
            if learner_name == "KernelSVM(rbf)":
                return SklearnWrapper("SVM", lambda: SVC(kernel="rbf"))
            return SklearnWrapper("Tree", lambda: DecisionTreeClassifier(max_depth=6, random_state=int(seed)))
        with st.spinner("Training and evaluating on an independent sample…"):
            t0 = time.perf_counter()
            rng = np.random.default_rng(int(seed))
            X, y = problem.sample(n, rng=rng)
            Xt, yt = problem.sample(1000, rng=rng)
            model = factory().fit(X, y)
            errors = int(np.sum(model.predict(Xt) != yt))
            train_error = float(np.mean(model.predict(X) != y))
            # GF2 only fits consistent parity; random labels need not be consistent.
            rad, li = None, None
            if learner_name != "GF2 exact parity":
                def fit_predict(Xtr, ytr, Xq):
                    return factory().fit(Xtr, ytr).predict(Xq)
                rad = empirical_rademacher(fit_predict, X, n_draws=3, rng=rng)
                li = learnability_index(errors/len(yt), errors/len(yt)-train_error, rad)
            st.session_state.result = dict(problem=problem.name, learner=learner_name, d=d, k=k,
                n=n, seed=int(seed), test_error=errors/len(yt), train_error=train_error,
                test_error_ci=wilson_interval(errors, len(yt)), random_label_correlation=rad,
                posthoc_li=li, updates=getattr(model, "updates_", 0),
                majority_test_error=float(min(np.mean(yt==1), np.mean(yt==-1))),
                seconds=time.perf_counter()-t0, theory=problem.theory)
    if "result" in st.session_state:
        r = st.session_state.result
        st.caption(f"Result: {r['learner']} on {r['problem']}, n={r['n']}, seed={r['seed']}")
        cols = st.columns(3)
        cols[0].metric("Test error", f"{r['test_error']:.3f}")
        cols[1].metric("Train error", f"{r['train_error']:.3f}")
        cols[2].metric("Optimizer updates", r['updates'])
        st.write(f"95% Wilson error interval: [{r['test_error_ci'][0]:.3f}, {r['test_error_ci'][1]:.3f}]. Runtime: {r['seconds']:.2f}s.")
        st.caption("Majority-label test reference (descriptive): " + f"{r['majority_test_error']:.3f}. Chance is not always 0.5 for imbalanced decision lists.")
        if r['posthoc_li'] is not None:
            st.write(f"Exploratory post-hoc LI: {r['posthoc_li']:.3f}; random-label fitting correlation: {r['random_label_correlation']:.3f}.")
        st.download_button("Download experiment JSON", json.dumps(r, indent=2), "learnlens-experiment.json", "application/json")
        with st.expander("Problem assumptions"):
            st.json(r['theory'])
with evidence:
    path = ROOT / "outputs/benchmark.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        st.subheader("Fixed-update SGD vs exact Gaussian elimination")
        st.write("Each SGD fit receives the same update budget. Sample grids use nested training data; validation selects crossings and an independent test set reports error.")
        st.caption(f"Generated {data['created_utc']} · {len(data['records'])} fits · config {data['config_sha256'][:12]}")
        frame = pd.DataFrame(data['records'])
        selected_d = st.selectbox("Benchmark dimension", sorted(frame.d.unique().tolist()))
        selected_k = st.selectbox("Benchmark sparsity", sorted(frame.k.unique().tolist()))
        subset = frame[(frame.d == selected_d) & (frame.k == selected_k)]
        means = subset.groupby(['n', 'learner']).err_test.mean().unstack()
        st.line_chart(means, x_label="Training samples", y_label="Mean test error")
        st.dataframe(subset[['n','seed','learner','err_test','test_error_ci','updates','seconds']], hide_index=True)
        st.write("Validation crossing intervals (upper error interval ≤ target):")
        st.dataframe(pd.DataFrame([r for r in data['transitions'] if r['d']==selected_d and r['k']==selected_k]), hide_index=True)
        st.download_button("Download benchmark + provenance", path.read_bytes(), "benchmark.json", "application/json")
    else:
        st.info("Run python run_benchmark.py to generate the audited benchmark.")
with methods:
    st.markdown("""
- **LI is post-hoc:** it includes test error and cannot predict learnability before training. Its weights are heuristic. A learning-curve exponent is reported separately in the library.
- **Random-label correlation** depends on the fitting algorithm; it is not a certified empirical Rademacher supremum.
- **Leap complexity** uses union of previously discovered monomial supports. The provided Fourier helper queries the target oracle; ordinary labeled datasets do not supply this access.
- **Centered NTK alignment** is available in the library to check sensitivity to label imbalance. Finite-width alignment is a diagnostic, not a theorem about SGD.
- **PAC-Bayes** now corrects for prior-grid selection and Monte Carlo error, and bounds a randomized Gibbs classifier. Earlier saved PAC-Bayes numbers are superseded.
- **Censoring:** no crossing by the largest n is right-censored; crossing at the smallest n is left-censored. Neither is an exact observed transition.
- **Scope:** noiseless synthetic tasks, a small network and few seeds. A finite failed training run does not establish computational hardness or guarantee an admissions outcome.
""")
    st.link_button("Merged-staircase property (COLT 2022)", "https://proceedings.mlr.press/v178/abbe22a.html")
    st.link_button("Leap complexity (COLT 2023)", "https://proceedings.mlr.press/v195/abbe23a.html")
st.caption("LearnLens · Reproducible experiments, explicit assumptions, downloadable evidence")
