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


@pytest.mark.network
def test_predict_streams_and_reports_accuracy(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setenv("TPV_SEED", "42")
    train.train(1, 14)
    capsys.readouterr()  # clear train output

    from tpv import predict
    acc = predict.predict(1, 14)
    out = capsys.readouterr().out
    assert "epoch 00:" in out
    assert "Accuracy:" in out
    assert 0.0 <= acc <= 1.0


@pytest.mark.network
def test_predict_per_chunk_latency_under_2s(tmp_path, monkeypatch):
    import time, joblib
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setenv("TPV_SEED", "42")
    train.train(1, 14)
    art = joblib.load(train.artifact_path(1, 14))
    pipe, X_test = art["pipeline"], art["X_test"]
    t0 = time.perf_counter()
    pipe.predict(X_test[0:1])
    assert (time.perf_counter() - t0) < 2.0


def test_predict_missing_artifact_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    from tpv import predict
    with pytest.raises(FileNotFoundError):
        predict.predict(1, 14)


@pytest.mark.network
def test_run_all_single_subject_returns_grand_mean(monkeypatch, capsys):
    monkeypatch.setenv("TPV_SEED", "42")
    from tpv import evaluate
    grand = evaluate.run_all(subjects=[1], experiments=range(6))
    out = capsys.readouterr().out
    assert "experiment 0:" in out
    assert "Mean accuracy of 6 experiments" in out
    assert 0.0 <= grand <= 1.0


@pytest.mark.network
def test_compare_classifiers_returns_all(monkeypatch):
    from tpv import evaluate
    scores = evaluate.compare_classifiers(1, 3)
    assert set(scores.keys()) == {"lda", "svm", "logreg", "rf"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
