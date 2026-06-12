import os
import numpy as np
import pytest
import joblib
from tpv import train, config


@pytest.mark.network
def test_train_creates_artifact_and_returns_mean(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setenv("TPV_SEED", "42")
    mean = train.train(1, 14)            # run 14 -> exp3
    out = capsys.readouterr().out
    assert "cross_val_score:" in out
    assert 0.0 <= mean <= 1.0

    path = train.artifact_path(1, 14)
    assert path.exists()
    art = joblib.load(path)
    assert set(art.keys()) == {"pipeline", "X_test", "y_test", "meta"}
    assert art["meta"]["experiment"] == 3
    assert art["X_test"].shape[0] == art["y_test"].shape[0] > 0
