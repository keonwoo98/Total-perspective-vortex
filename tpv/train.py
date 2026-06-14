"""Train mode: cross_val_score report + fit + persist artifact."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split

from tpv import config, preprocessing
from tpv.pipeline import build_pipeline


def artifact_path(subject: int, run: int) -> Path:
    """Pure path query (no side effects). train() creates the parent dir."""
    exp = config.RUN_TO_EXPERIMENT[run]
    return config.MODELS_DIR / f"subj{subject:03d}_run{run:02d}_exp{exp}.joblib"


def train(subject: int, run: int) -> float:
    exp = config.RUN_TO_EXPERIMENT[run]
    seed = config.get_seed()
    X, y = preprocessing.build_dataset(subject, exp)

    pipe = build_pipeline(csp="scratch", clf="lda",
                          n_components=config.N_COMPONENTS, reg=config.REG, seed=seed)

    # cross_val_score over the whole pipeline (CSP refit per fold -> no leakage).
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv)
    print(np.array2string(scores, precision=4, floatmode="fixed"))
    print(f"cross_val_score: {scores.mean():.4f}")

    # Deployable model: hold out a test split that predict mode will replay.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    pipe.fit(X_tr, y_tr)

    artifact = {
        "pipeline": pipe,
        "X_test": X_te,
        "y_test": y_te,
        "meta": {"subject": subject, "run": run, "experiment": exp,
                 "tmin": config.TMIN, "tmax": config.TMAX,
                 "n_components": config.N_COMPONENTS},
    }
    path = artifact_path(subject, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return float(scores.mean())
