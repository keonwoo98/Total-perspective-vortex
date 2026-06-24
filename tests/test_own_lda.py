import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from tpv.own_lda import OwnLDA


def _csp_like(n=120, n_features=4, seed=0, balance=(0.5, 0.5)):
    rng = np.random.default_rng(seed)
    n1 = int(round(n * balance[1])); n0 = n - n1
    X0 = rng.standard_normal((n0, n_features))
    X1 = rng.standard_normal((n1, n_features)); X1[:, 0] += 2.0   # class 1 shifted on feature 0
    X = np.vstack([X0, X1]); y = np.array([0] * n0 + [1] * n1)
    return X, y


def test_ownlda_parity_with_sklearn():
    X, y = _csp_like(seed=1)
    own = OwnLDA().fit(X, y)
    skl = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
    assert abs((own.predict(X) == y).mean() - (skl.predict(X) == y).mean()) < 0.05


def test_ownlda_label_agreement_asymmetric():
    # unbalanced classes + non-{0,1} labels -> catches a reversed class mapping
    X, y01 = _csp_like(seed=2, balance=(0.7, 0.3))
    y = np.where(y01 == 0, 7, 9)
    own = OwnLDA().fit(X, y)
    skl = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
    assert (own.predict(X) == skl.predict(X)).mean() > 0.95
    assert set(own.predict(X)).issubset({7, 9})
    assert list(own.classes_) == [7, 9]


def test_ownlda_clone_and_shape():
    X, y = _csp_like(seed=3)
    assert clone(OwnLDA(reg=1e-2)).get_params() == {"reg": 1e-2}
    own = OwnLDA().fit(X, y)
    assert own.coef_.shape == (4,)


def test_ownlda_in_pipeline():
    X, y = _csp_like(seed=4)
    scores = cross_val_score(Pipeline([("clf", OwnLDA())]), X, y, cv=3)
    assert scores.mean() > 0.8
