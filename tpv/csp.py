"""From-scratch Common Spatial Patterns as a scikit-learn transformer.

Covariance estimation (numpy) + generalized eigendecomposition (scipy.linalg.eigh).
No SVD, no library CSP. See spec section 4.
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class MyCSP(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 4, reg: float = 0.01):
        # Store hyperparameters only (sklearn clone/set_params contract).
        self.n_components = n_components
        self.reg = reg

    def _class_cov(self, epochs: np.ndarray) -> np.ndarray:
        """Mean trace-normalized, symmetrized, shrinkage-regularized covariance."""
        n_ch = epochs.shape[1]
        acc = np.zeros((n_ch, n_ch))
        # numpy 2.x + macOS Accelerate emits spurious matmul RuntimeWarnings here;
        # the covariances are numerically correct, so silence the noise.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for E in epochs:
                C = E @ E.T
                tr = np.trace(C)
                if tr > 0:
                    C = C / tr
                acc += C
            C = acc / len(epochs)
            C = (C + C.T) / 2.0
            C = (1.0 - self.reg) * C + self.reg * (np.trace(C) / n_ch) * np.eye(n_ch)
        return C

    def fit(self, X, y=None):
        # y is required (CSP is supervised); the y=None default only satisfies the
        # sklearn transformer signature convention. Pipeline always passes y.
        if y is None:
            raise ValueError("y is required for supervised MyCSP.fit()")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("MyCSP requires exactly 2 classes")

        c1 = self._class_cov(X[y == self.classes_[0]])
        c2 = self._class_cov(X[y == self.classes_[1]])

        # numpy 2.x + macOS Accelerate emits spurious matmul RuntimeWarnings inside
        # pinv; the decomposition is numerically correct, so silence the noise.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            # Generalized eigenproblem: C1 w = lambda (C1+C2) w. eigh -> ascending eigenvalues.
            eigvals, eigvecs = eigh(c1, c1 + c2)
            order = np.argsort(np.abs(eigvals - 0.5))[::-1]  # most discriminative first
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            self.eigenvalues_ = eigvals
            self.filters_ = eigvecs.T[: self.n_components]          # (n_components, n_channels)
            self.patterns_ = np.linalg.pinv(eigvecs.T)[:, : self.n_components]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((X.shape[0], self.n_components))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for i, E in enumerate(X):
                Z = self.filters_ @ E                # (n_components, n_times)
                var = np.var(Z, axis=1)
                total = var.sum()
                var = var / total if total > 0 else np.full_like(var, 1.0 / var.size)
                out[i] = np.log(np.clip(var, 1e-12, None))   # guard log(0) on degenerate input
        return out
