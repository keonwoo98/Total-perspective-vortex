"""Factory for the CSP + classifier scikit-learn Pipeline.

CSP is the FIRST step so cross_val_score refits it per training fold (no leakage).
"""
from mne.decoding import CSP as MneCSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from tpv.csp import MyCSP
from tpv.own_lda import OwnLDA


def _make_classifier(clf: str, seed):
    if clf == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if clf == "svm":
        return SVC(kernel="rbf", C=1.0, random_state=seed)
    if clf == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if clf == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed)
    if clf == "own-lda":
        return OwnLDA()
    raise ValueError(f"unknown classifier: {clf}")


def _make_csp(csp: str, n_components: int, reg: float):
    if csp == "scratch":
        return MyCSP(n_components=n_components, reg=reg)
    if csp == "mne":
        # Close to MyCSP's route for the parity test; the two still differ by MyCSP's
        # 0.01 identity shrinkage and component ordering, absorbed by the <0.05
        # tolerance. If parity ever fails, first match regularization (set reg=0.01),
        # do NOT relax the tolerance (spec section 4.6).
        return MneCSP(n_components=n_components, reg=None, log=True,
                      norm_trace=True, cov_est="epoch")
    if csp == "fbcsp":
        from tpv.fbcsp import FilterBankCSP
        return FilterBankCSP(n_components=2, reg=reg)
    raise ValueError(f"unknown csp: {csp}")


def build_pipeline(csp: str = "scratch", clf: str = "lda",
                   n_components: int = 4, reg: float = 0.01, seed=None) -> Pipeline:
    return Pipeline([
        ("csp", _make_csp(csp, n_components, reg)),
        ("clf", _make_classifier(clf, seed)),
    ])
