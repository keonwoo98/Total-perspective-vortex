import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from tpv.csp import MyCSP
from tpv.pipeline import build_pipeline
from mne.decoding import CSP as MneCSP


def _synthetic(n_per_class=40, n_ch=8, n_times=160, seed=0):
    """Two classes: class 0 has extra variance in channel 0, class 1 in channel 1."""
    rng = np.random.default_rng(seed)
    base0 = rng.standard_normal((n_per_class, n_ch, n_times))
    base1 = rng.standard_normal((n_per_class, n_ch, n_times))
    base0[:, 0, :] *= 4.0
    base1[:, 1, :] *= 4.0
    X = np.concatenate([base0, base1], axis=0)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def test_fit_sets_learned_attributes():
    X, y = _synthetic()
    csp = MyCSP(n_components=4).fit(X, y)
    assert csp.filters_.shape == (4, X.shape[1])
    assert set(csp.classes_.tolist()) == {0, 1}


def test_transform_output_shape():
    X, y = _synthetic()
    feats = MyCSP(n_components=4).fit(X, y).transform(X)
    assert feats.shape == (X.shape[0], 4)
    assert np.isfinite(feats).all()


def test_eigenvalues_in_unit_interval():
    X, y = _synthetic()
    csp = MyCSP(n_components=4).fit(X, y)
    assert csp.eigenvalues_.min() >= -1e-9
    assert csp.eigenvalues_.max() <= 1 + 1e-9


def test_csp_separates_synthetic_classes():
    X, y = _synthetic(seed=1)
    pipe = Pipeline([("csp", MyCSP(n_components=4)),
                     ("clf", LinearDiscriminantAnalysis())])
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.95


def test_clone_compatibility():
    csp = MyCSP(n_components=6, reg=0.05)
    c2 = clone(csp)
    assert c2.get_params() == {"n_components": 6, "reg": 0.05, "solver": "eigh"}


def test_build_pipeline_default_is_scratch_lda():
    pipe = build_pipeline()
    assert list(dict(pipe.steps).keys()) == ["csp", "clf"]
    assert isinstance(pipe.named_steps["csp"], MyCSP)
    assert isinstance(pipe.named_steps["clf"], LinearDiscriminantAnalysis)


def test_build_pipeline_mne_csp_option():
    pipe = build_pipeline(csp="mne")
    assert isinstance(pipe.named_steps["csp"], MneCSP)


def test_build_pipeline_classifier_choices():
    for clf in ("lda", "svm", "logreg", "rf"):
        pipe = build_pipeline(clf=clf)
        assert pipe.named_steps["clf"] is not None
