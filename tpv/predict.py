"""Predict mode: replay persisted held-out epochs as a simulated stream (<2s/chunk)."""
import time
import joblib
import numpy as np

from tpv import config
from tpv.train import artifact_path

LATENCY_BUDGET_S = 2.0


def predict(subject: int, run: int) -> float:
    path = artifact_path(subject, run)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `train {subject} {run}` first.")

    art = joblib.load(path)                       # heavy work BEFORE the loop
    pipe, X_test, y_test = art["pipeline"], art["X_test"], art["y_test"]

    print("epoch nb: [prediction] [truth] equal?")
    correct = 0
    for i in range(X_test.shape[0]):
        chunk = X_test[i:i + 1]                   # (1, 64, n_times)
        t0 = time.perf_counter()
        pred = int(pipe.predict(chunk)[0])        # CSP transform + LDA dot product
        latency = time.perf_counter() - t0
        assert latency < LATENCY_BUDGET_S, f"chunk {i} took {latency:.3f}s"

        truth = int(y_test[i])
        equal = pred == truth
        correct += equal
        # Map internal class index {0,1} -> {1,2} for parity with the PDF example.
        # (predict is single-run addressable -> only exp0-3, where 0/1 == T1/T2.)
        print(f"epoch {i:02d}:    [{pred + 1}]    [{truth + 1}] {equal}")

    acc = correct / X_test.shape[0] if X_test.shape[0] else 0.0
    print(f"Accuracy: {acc:.4f}")
    return float(acc)
