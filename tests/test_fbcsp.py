import numpy as np
import pytest
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from tpv.fbcsp import FilterBankCSP, DEFAULT_BANDS


def _synthetic(n_per_class=30, n_ch=8, n_times=321, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_per_class, n_ch, n_times)); a[:, 0] *= 4.0
    b = rng.standard_normal((n_per_class, n_ch, n_times)); b[:, 1] *= 4.0
    X = np.concatenate([a, b])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def test_fbcsp_transform_shape():
    X, y = _synthetic()
    feats = FilterBankCSP(n_components=2).fit(X, y).transform(X)
    assert feats.shape == (X.shape[0], len(DEFAULT_BANDS) * 2)
    assert np.isfinite(feats).all()


def test_fbcsp_in_pipeline():
    X, y = _synthetic(seed=1)
    pipe = Pipeline([("csp", FilterBankCSP(n_components=2)),
                     ("clf", LinearDiscriminantAnalysis())])
    scores = cross_val_score(pipe, X, y, cv=3)
    assert np.isfinite(scores).all()


def test_fbcsp_clone():
    assert isinstance(clone(FilterBankCSP(n_components=2)), FilterBankCSP)


def test_fbcsp_epoch_length_guard():
    X, y = _synthetic(n_times=100)   # too short for the 8 Hz band FIR
    with pytest.raises(ValueError):
        FilterBankCSP().fit(X, y)
