"""Cross-validated accuracy, classifier comparison, and the full 109x6 run."""
import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_val_score

from tpv import config, preprocessing
from tpv.pipeline import build_pipeline

CLASSIFIERS = ("lda", "svm", "logreg", "rf")


def tune(subject: int, experiment: int, grid=None):
    """Opt-in hyperparameter tuning via GridSearchCV over the whole pipeline (no leakage:
    CSP is refit inside each inner fold). Returns (best_params_, best_score_) and prints them.
    Demo-only — run_all stays on fixed config.N_COMPONENTS."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    if grid is None:
        grid = {"csp__n_components": [4, 6, 8]}
    X, y = preprocessing.build_dataset(subject, experiment)
    pipe = build_pipeline(csp="scratch", clf="lda")
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # fixed for a stable demo
    search = GridSearchCV(pipe, grid, cv=inner, scoring="accuracy")
    search.fit(X, y)
    print(f"best params: {search.best_params_}   best score: {search.best_score_:.4f}")
    return search.best_params_, search.best_score_


def cross_val_accuracy(subject: int, experiment: int, csp="scratch", clf="lda") -> float:
    seed = config.get_seed()
    X, y = preprocessing.build_dataset(subject, experiment)
    pipe = build_pipeline(csp=csp, clf=clf,
                          n_components=config.N_COMPONENTS, reg=config.REG, seed=seed)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=seed)
    return float(cross_val_score(pipe, X, y, cv=cv).mean())


def compare_classifiers(subject: int, experiment: int) -> dict[str, float]:
    return {clf: cross_val_accuracy(subject, experiment, clf=clf) for clf in CLASSIFIERS}


def run_all(subjects=range(1, config.N_SUBJECTS + 1), experiments=range(6), fast=False) -> float:
    subjects = list(subjects)
    experiments = list(experiments)
    per_exp_means = []
    for exp in experiments:
        accs = []
        skipped = 0
        for subj in subjects:
            try:
                acc = cross_val_accuracy(subj, exp)
            # Narrow catch: only DATA/IO failures are skippable. Real algorithm bugs
            # (ValueError, KeyError, LinAlgError, ...) must propagate, not be hidden.
            except (RuntimeError, FileNotFoundError, OSError) as e:
                print(f"  [skip] subject {subj:03d} exp {exp}: {type(e).__name__}: {e}")
                skipped += 1
                continue
            accs.append(acc)
            if fast:
                print(f"experiment {exp}: subject {subj:03d}: accuracy = {acc:.4f}")
        if not accs:
            raise RuntimeError(f"experiment {exp}: all {len(subjects)} subjects failed to evaluate")
        mean = float(np.mean(accs))
        per_exp_means.append(mean)
        print(f"experiment {exp}:    accuracy = {mean:.4f}   (n={len(accs)}, skipped={skipped})")
    grand = float(np.mean(per_exp_means)) if per_exp_means else 0.0
    print(f"Mean accuracy of 6 experiments: {grand:.4f}")
    return grand
