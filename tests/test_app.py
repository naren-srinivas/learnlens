"""Execute the real Streamlit script and all supported learner/problem paths."""
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'


@pytest.mark.parametrize('problem,learner',[
    ('k-sparse parity','LogisticRegression'),
    ('k-sparse parity','KernelSVM(rbf)'),
    ('k-sparse parity','DecisionTree'),
    ('k-sparse parity','NumpyMLP'),
    ('k-sparse parity','GF2 exact parity'),
    ('margin halfspace','LogisticRegression'),
    ('margin halfspace','KernelSVM(rbf)'),
    ('margin halfspace','DecisionTree'),
    ('margin halfspace','NumpyMLP'),
    ('k-decision list','LogisticRegression'),
    ('k-decision list','KernelSVM(rbf)'),
    ('k-decision list','DecisionTree'),
    ('k-decision list','NumpyMLP'),
])
def test_app_executes_experiment(problem,learner):
    app = AppTest.from_file(APP,default_timeout=60).run()
    assert not app.exception
    next(w for w in app.selectbox if w.label == 'Problem').select(problem).run()
    next(w for w in app.selectbox if w.label == 'Learner').select(learner)
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state['result']
    assert 0 <= result['test_error'] <= 1
    assert result['learner'] == learner


def test_k_control_never_exceeds_dimension():
    app = AppTest.from_file(APP,default_timeout=30).run()
    next(w for w in app.slider if w.label == 'Ambient dimension d').set_value(4).run()
    assert next(w for w in app.slider if w.label == 'Sparsity / list length k').max == 4


@pytest.mark.parametrize('learner',['LogisticRegression','KernelSVM(rbf)'])
def test_single_class_decision_list_uses_explicit_fallback(learner):
    app = AppTest.from_file(APP,default_timeout=60).run()
    next(w for w in app.selectbox if w.label == 'Problem').select('k-decision list').run()
    next(w for w in app.slider if w.label == 'Ambient dimension d').set_value(8).run()
    next(w for w in app.slider if w.label == 'Sparsity / list length k').set_value(8).run()
    next(w for w in app.select_slider if w.label == 'Training samples n').set_value(50)
    next(w for w in app.selectbox if w.label == 'Learner').select(learner)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state['result']['constant_class_fallback']
    assert any('only one label' in item.value for item in app.info)
