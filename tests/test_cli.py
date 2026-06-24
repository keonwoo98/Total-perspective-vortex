import pytest
import mybci


def test_no_args_calls_run_all(monkeypatch):
    called = {}
    monkeypatch.setattr(mybci, "run_all", lambda: called.setdefault("ran", True))
    assert mybci.main([]) == 0
    assert called.get("ran")


def test_train_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(mybci, "train", lambda s, r: seen.update(s=s, r=r))
    assert mybci.main(["4", "14", "train"]) == 0
    assert seen == {"s": 4, "r": 14}


def test_predict_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(mybci, "predict", lambda s, r: seen.update(s=s, r=r))
    assert mybci.main(["4", "14", "predict"]) == 0
    assert seen == {"s": 4, "r": 14}


def test_invalid_subject_errors():
    assert mybci.main(["200", "14", "train"]) != 0


def test_invalid_run_errors():
    assert mybci.main(["4", "2", "train"]) != 0     # run 2 is baseline, not a task


def test_bad_arity_errors():
    assert mybci.main(["4", "14"]) != 0             # missing mode


def test_tune_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(mybci, "tune", lambda s, e: seen.update(s=s, e=e))
    assert mybci.main(["4", "14", "tune"]) == 0
    assert seen == {"s": 4, "e": 3}   # run 14 -> experiment 3
