"""From-scratch two-class Linear Discriminant Analysis (sklearn-compatible).

numpy + np.linalg.solve only. Drop-in for Pipeline([CSP, OwnLDA]) via build_pipeline(clf="own-lda").
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class OwnLDA(ClassifierMixin, BaseEstimator):
    def __init__(self, reg: float = 1e-3):
        # Store hyperparameters only (sklearn clone/set_params contract).
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("OwnLDA supports exactly 2 classes")
        self.n_features_in_ = X.shape[1]

        # Invariant: mu0 <-> classes_[0], mu1 <-> classes_[1]; w points toward class 1.
        X0 = X[y == self.classes_[0]]
        X1 = X[y == self.classes_[1]]
        mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
        p0, p1 = X0.shape[0] / X.shape[0], X1.shape[0] / X.shape[0]

        # Priors-weighted pooled within-class covariance (biased, matches sklearn scaling),
        # ridge-regularized so it stays invertible on few CSP features.
        sigma = p0 * np.cov(X0, rowvar=False, bias=True) + p1 * np.cov(X1, rowvar=False, bias=True)
        sigma = sigma + self.reg * np.eye(self.n_features_in_)

        self.coef_ = np.linalg.solve(sigma, mu1 - mu0)          # w = Sigma^-1 (mu1 - mu0)
        self.intercept_ = -self.coef_ @ (mu0 + mu1) / 2.0 + np.log(p1 / p0)
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict(self, X):
        # decision > 0 -> class 1 (follows from w = Sigma^-1 (mu1 - mu0))
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
