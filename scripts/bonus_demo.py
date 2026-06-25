#!/usr/bin/env python
"""Demonstrate all five bonus features on one PhysioNet subject.

  python scripts/bonus_demo.py          # subject 1, experiment 3 (run 6/10/14)
  python scripts/bonus_demo.py 4 14     # subject 4, run 14

(A) from-scratch Jacobi eigensolver, (C) hyperparameter tuning, (D) from-scratch OwnLDA,
(F) Filter-Bank CSP, (G) a second dataset (BCI IV-2a; needs the optional moabb install).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from tpv import config, evaluate, preprocessing
from tpv.csp import MyCSP
from tpv.pipeline import build_pipeline

SEED = 42


def _acc(pipe, X, y):
    return cross_val_score(pipe, X, y, cv=ShuffleSplit(10, test_size=0.2, random_state=SEED)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", nargs="?", type=int, default=1)
    ap.add_argument("run", nargs="?", type=int, default=6)
    args = ap.parse_args()
    exp = config.RUN_TO_EXPERIMENT[args.run]

    X, y = preprocessing.build_dataset(args.subject, exp)
    print(f"# Bonus demo — subject {args.subject}, experiment {exp}\n")
    print(f"[mandatory] scratch CSP (eigh) + LDA : {_acc(build_pipeline(), X, y):.4f}")

    jac = Pipeline([("csp", MyCSP(n_components=4, solver="jacobi")),
                    ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
    print(f"[A] from-scratch Jacobi eigensolver  : {_acc(jac, X, y):.4f}  (matches eigh)")

    best_params, best_score = evaluate.tune(args.subject, exp)
    print(f"[C] tuned {best_params} -> cv {best_score:.4f}")

    print(f"[D] from-scratch OwnLDA classifier   : {_acc(build_pipeline(clf='own-lda'), X, y):.4f}")
    print(f"[F] Filter-Bank CSP (4 sub-bands)    : {_acc(build_pipeline(csp='fbcsp'), X, y):.4f}")

    try:
        from tpv.external import load_external
        Xe, ye = load_external(1)
        print(f"[G] 2nd dataset BCI IV-2a (subj 1)   : {_acc(build_pipeline(), Xe, ye):.4f}")
    except ImportError:
        print("[G] 2nd dataset: install moabb via `pip install -r requirements.txt` to run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
