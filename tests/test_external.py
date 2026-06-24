import numpy as np
import pytest
from sklearn.model_selection import ShuffleSplit, cross_val_score
from tpv.pipeline import build_pipeline


def test_external_pipeline_shape_compat():
    # No moabb needed: proves the existing pipeline accepts a 22-channel / 250 Hz-like dataset
    # (MyCSP is channel-count-agnostic), which is what load_external returns.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((20, 22, 500)); a[:, 0] *= 4.0
    b = rng.standard_normal((20, 22, 500)); b[:, 1] *= 4.0
    X = np.concatenate([a, b]); y = np.array([0] * 20 + [1] * 20)
    scores = cross_val_score(build_pipeline(csp="scratch", clf="lda"), X, y, cv=3)
    assert np.isfinite(scores).all()


@pytest.mark.network
def test_load_external_smoke():
    pytest.importorskip("moabb")
    from tpv.external import load_external
    X, y = load_external(1)
    assert X.ndim == 3
    assert set(np.unique(y)).issubset({0, 1})
    acc = cross_val_score(build_pipeline(), X, y,
                          cv=ShuffleSplit(5, test_size=0.2, random_state=42)).mean()
    print(f"BNCI2014_001 subject 1 left-vs-right accuracy: {acc:.4f}")  # observation, not an assert
