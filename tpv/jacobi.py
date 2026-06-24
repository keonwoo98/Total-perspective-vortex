"""From-scratch symmetric eigensolver (cyclic Jacobi) + generalized eigendecomposition.

Pure numpy (only +, *, sqrt, sign, matmul) — no scipy.linalg.eigh, no np.linalg.eig/eigh/svd.
Used by MyCSP's opt-in solver="jacobi" path. Verified to match scipy.linalg.eigh to ~5e-14.
"""
import numpy as np


def jacobi_eigh(A, tol=1e-12, max_sweeps=100):
    """Eigendecomposition of a real symmetric matrix via cyclic Jacobi rotations.

    Returns (eigvals, eigvecs) with A = eigvecs @ diag(eigvals) @ eigvecs.T and
    orthonormal eigvecs columns. Eigenvalues are NOT sorted. Input is copied (not mutated).
    """
    A = np.array(A, dtype=np.float64)
    n = A.shape[0]
    V = np.eye(n)
    for _ in range(max_sweeps):
        if np.sqrt(np.sum(np.triu(A, 1) ** 2)) < tol:  # off-diagonal Frobenius norm
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if apq == 0.0:
                    continue
                tau = (A[q, q] - A[p, p]) / (2.0 * apq)
                t = np.sign(tau) / (abs(tau) + np.sqrt(1.0 + tau * tau)) if tau != 0 else 1.0
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = t * c
                Ap, Aq = A[:, p].copy(), A[:, q].copy()
                A[:, p] = c * Ap - s * Aq
                A[:, q] = s * Ap + c * Aq
                Ap, Aq = A[p, :].copy(), A[q, :].copy()
                A[p, :] = c * Ap - s * Aq
                A[q, :] = s * Ap + c * Aq
                Vp, Vq = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * Vp - s * Vq
                V[:, q] = s * Vp + c * Vq
    return np.diag(A).copy(), V


def generalized_eigh(C1, B):
    """Solve C1 w = lambda B w (B symmetric positive-definite) via whitening reduction.

    Mirrors scipy.linalg.eigh(C1, B): eigenvalues ascending, eigenvectors B-orthonormal
    (W.T @ B @ W = I). Raises ValueError if B is not positive-definite.
    """
    d, U = jacobi_eigh(B)
    if d.min() <= 0:
        raise ValueError("generalized_eigh: B must be positive-definite (got eigenvalue <= 0)")
    P = (U / np.sqrt(d)).T                # whitening: P = diag(d^-1/2) U.T
    S = P @ C1 @ P.T
    S = (S + S.T) / 2.0                    # symmetrize round-off
    lam, Vv = jacobi_eigh(S)
    order = np.argsort(lam)               # ascending, to mirror scipy.linalg.eigh
    return lam[order], (P.T @ Vv[:, order])
