import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from scipy.linalg import eigh
from tpv.csp import MyCSP
from tpv.jacobi import jacobi_eigh, generalized_eigh


def _synthetic(n_per_class=40, n_ch=8, n_times=160, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_per_class, n_ch, n_times)); a[:, 0] *= 4.0
    b = rng.standard_normal((n_per_class, n_ch, n_times)); b[:, 1] *= 4.0
    X = np.concatenate([a, b])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def test_jacobi_eigh_matches_numpy():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((10, 10)); A = M + M.T
    w, V = jacobi_eigh(A)
    assert np.allclose(np.sort(w), np.linalg.eigvalsh(A), atol=1e-9)
    assert np.allclose(V @ np.diag(w) @ V.T, A, atol=1e-9)


def test_generalized_eigh_matches_scipy():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((8, 8)); C1 = M @ M.T + np.eye(8)
    N = rng.standard_normal((8, 8)); B = C1 + (N @ N.T + np.eye(8))
    lam, W = generalized_eigh(C1, B)
    lam_ref, _ = eigh(C1, B)
    assert np.allclose(lam, lam_ref, atol=1e-9)
    assert np.allclose(W.T @ B @ W, np.eye(8), atol=1e-9)


def test_mycsp_jacobi_matches_eigh():
    X, y = _synthetic(seed=2)
    csp_e = MyCSP(n_components=4, solver="eigh").fit(X, y)
    csp_j = MyCSP(n_components=4, solver="jacobi").fit(X, y)
    assert np.allclose(np.sort(csp_e.eigenvalues_), np.sort(csp_j.eigenvalues_), atol=1e-9)
    assert csp_j.eigenvalues_.min() >= -1e-9 and csp_j.eigenvalues_.max() <= 1 + 1e-9
    for fe, fj in zip(csp_e.filters_, csp_j.filters_):
        cos = abs(fe @ fj) / (np.linalg.norm(fe) * np.linalg.norm(fj))
        assert cos > 0.999


def test_jacobi_pipeline_runs():
    X, y = _synthetic(seed=3)
    pipe = Pipeline([("csp", MyCSP(n_components=4, solver="jacobi")),
                     ("clf", LinearDiscriminantAnalysis())])
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.9


def test_generalized_eigh_pd_guard():
    with pytest.raises(ValueError):
        generalized_eigh(np.eye(3), np.diag([1.0, 0.0, -1.0]))
