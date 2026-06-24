# Total Perspective Vortex — Bonus Design Spec (verified)

> Implements the bonus on top of the completed mandatory project. Goal: tick all three
> evaluation bonus boxes — **Implementations** (A jacobi eigensolver, C hyperparameter tuning,
> D own classifier), **Feature engineering** (F minimal FBCSP), **Datasets** (G second dataset) —
> SAFELY, with CLEAN CODE and MINIMAL added complexity.
>
> Date: 2026-06-24. Researched + adversarially verified (14-agent workflow).

## Guiding principles (apply to every item)

1. **The graded mandatory path is frozen.** `build_pipeline(csp="scratch", clf="lda", n_components=4,
   reg=0.01)`, `MyCSP`'s `scipy.linalg.eigh` route, and `run_all` stay byte-identical. Baseline:
   `pytest -m "not network"` = 21 passed; full 109×6 grand mean ≈ 0.658 ≥ 0.60. **No default changes.**
2. **Opt-in over replacement.** Every bonus is reached through an existing seam — a new string branch
   in `_make_csp`/`_make_classifier`, or one new hyperparameter with a mandatory-preserving default.
   No new abstractions, ABCs, or plugin framework.
3. **One small single-responsibility module per item**, numpy-only where possible, mirroring
   `tpv/csp.py` house style (module docstring; store-params-only `__init__`; trailing-underscore
   fitted attrs; `np.errstate` guard for macOS Accelerate matmul noise).
4. **Parity tests, not `check_estimator`.** Each from-scratch component is validated by asserting
   near-parity (features/accuracy/labels) against its library counterpart on the CSP feature space.
5. **No new hard dependencies.** The only heavy dependency (moabb, item G) is quarantined in
   `requirements-bonus.txt` and lazy-imported; verification done in a throwaway `.venv-bonus`.

---

## ITEM A — From-scratch eigensolver (cyclic Jacobi + whitening)

**Module:** `tpv/jacobi.py` (~45 lines, numpy only — only `+,*,sqrt,sign,@`; no eig/eigh/svd):
- `jacobi_eigh(A, tol=1e-12, max_sweeps=100)` — cyclic Jacobi for a real symmetric matrix; returns
  `(eigvals, eigvecs)` (eigvals unsorted, eigvecs orthonormal columns). Copies input
  (`np.array(A, float64)`) so it never mutates the caller's covariance.
- `generalized_eigh(C1, B)` — whitening reduction: `B = U diag(d) Uᵀ` → `P = diag(d^-1/2) Uᵀ`
  → `S = P C1 Pᵀ`, symmetrize `S=(S+S.T)/2`, `S = V diag(λ) Vᵀ`, return `λ` **ascending** and
  `W = Pᵀ V`. Reproduces `scipy.linalg.eigh(C1, B)` type=1 exactly (verified to 5.4e-14:
  eigenvalues match <1e-10, filters match up to per-row sign, `Wᵀ B W = I`).

**Integration:** `MyCSP.__init__` gains `solver="eigh"` (default) | `"jacobi"` (store only). `fit()`
branches inside the **existing `np.errstate` block**:
`eigvals, eigvecs = eigh(c1, c1+c2) if self.solver == "eigh" else generalized_eigh(c1, c1+c2)`.
Everything downstream (`argsort(abs(eigvals-0.5))[::-1]`, `filters_`, `patterns_`, `transform`)
is untouched.

**Opt-in (default `eigh`).** Verified byte-equivalent, so defaulting to jacobi would also be safe,
but opt-in removes all regression risk to the 0.658 gate. We additionally verify the jacobi path
clears 60% once, as proof.

**Tests** (`tests/test_jacobi.py`): (1) `jacobi_eigh` vs `np.linalg.eigh` on a random symmetric
matrix (eigvals <1e-10); (2) `MyCSP(solver="jacobi")` vs `MyCSP(solver="eigh")` on `_synthetic` —
filters match up to per-row sign (`|cos|>0.999`), eigvals <1e-10, eigvals ∈ [0,1].

---

## ITEM C — Hyperparameter tuning (leakage-free, opt-in)

**Function:** `tune(subject, experiment, grid=None)` (~12 lines) in `tpv/evaluate.py`. Wrap the whole
`Pipeline([csp, clf])` in `GridSearchCV` over a tiny grid (default `{"csp__n_components": [4, 6, 8]}`)
with `StratifiedKFold(5, shuffle=True, random_state=42)` (fixed for demo determinism). CSP is step 0
so GridSearchCV refits it per inner fold → **no leakage** (verified: fit-count = grid×folds + 1).
Returns and prints `(best_params_, best_score_)`.

**CLI:** opt-in mode `python mybci.py <subject> <run> tune` — **3 args, so the existing arity guard is
not broken**; routes `run`→experiment via `RUN_TO_EXPERIMENT`. (exp4/5 are not reachable from a single
run — documented; tune covers exp0–3.) `run_all`/`config.N_COMPONENTS=4` stay fixed → 60% gate untouched.

**Tests:** one smoke test asserting `tune()` returns `best_params_` with a `csp__n_components` key on
small synthetic input. `mybci.py` `tune` dispatch covered by updated `tests/test_cli.py`.

---

## ITEM D — Own LDA classifier (`OwnLDA`)

**Module:** `tpv/own_lda.py` (~35 lines, numpy + `np.linalg.solve`). `class OwnLDA(ClassifierMixin,
BaseEstimator)` (**mixin first**). `fit(X, y)`: `self.classes_ = np.unique(y)`; **`mu0` = mean of
`classes_[0]`, `mu1` = mean of `classes_[1]`** (load-bearing invariant); pooled within-class covariance
(`bias=True`, priors-weighted to match sklearn scaling) + `reg*I` (reg=1e-3); `w = Σ⁻¹(mu1-mu0)` via
`np.linalg.solve`; `b = -w·(mu0+mu1)/2 + log(p1/p0)`; set `coef_`, `intercept_`, `n_features_in_`.
`predict(X)` returns `self.classes_[(decision_function(X) > 0).astype(int)]` — **original labels**.

**Integration:** `pipeline.py` `_make_classifier` gains `if clf == "own-lda": return OwnLDA()`. **Never
added to `evaluate.CLASSIFIERS`** (keeps it off the run_all default path). Default `clf="lda"` unchanged.

**Tests** (`tests/test_own_lda.py`): parity vs `LinearDiscriminantAnalysis(solver="lsqr",
shrinkage="auto")` on CSP-like 4-feature data — accuracy diff <0.05 AND **label agreement on
asymmetric/unbalanced synthetic data** (so a reversed class mapping cannot pass); `classes_` holds
original labels; `coef_.shape == (n_features,)`; predictions ⊆ `classes_`. Not `check_estimator`.

---

## ITEM F — Minimal FBCSP

**Module:** `tpv/fbcsp.py` (~50 lines). `class FilterBankCSP(TransformerMixin, BaseEstimator)` holding
K `MyCSP`. `__init__(bands=DEFAULT_BANDS, sfreq=config.SFREQ, n_components=2, reg=config.REG)`
(store only). `fit(X, y)`: for each of 4 fixed sub-bands `(8-12, 12-16, 16-20, 20-30 Hz)` (strict
subsets of the existing 7-30 Hz passband), band-pass X, fit a fresh `MyCSP(n_components, reg)`, store
in `self.csps_`. `transform(X)`: re-band-pass per band, `MyCSP.transform` each, `np.concatenate(axis=1)`
→ `(n, 4*2) = (n, 8)`. **No MIBIF, no FFT path.**

**Band-pass:** `mne.filter.filter_data(X, sfreq, lo, hi, method="fir", verbose="ERROR")` (FIR default —
verified no warnings; consistent with `preprocessing.filter_raw`). **Band-pass lives inside
`fit`/`transform`** so cross_val_score refits per fold (no leakage). No `build_dataset_wide` needed.

**Integration:** `pipeline.py` `_make_csp` gains `if csp == "fbcsp": from tpv.fbcsp import
FilterBankCSP; return FilterBankCSP(n_components=2, reg=reg)` (local import). Kept out of `run_all`.

**Tests** (`tests/test_fbcsp.py`): `fit/transform` shape `(n, 8)`, all-finite, runs inside
`Pipeline + cross_val_score`, `clone()` succeeds. Honest expectation: modest gain (~+0.01–0.04).

---

## ITEM G — Second dataset (highest-risk; safest viable path)

**Module:** `tpv/external.py` (~35 lines) exposing `load_external(subject=1) -> (X, y)` in the same
`(n_epochs, n_channels, n_times)` float64 / int{0,1} contract as `build_dataset`. Uses **moabb +
`BNCI2014_001`** (BCI Competition IV-2a) restricted to **left-vs-right-hand** via `LeftRightImagery`
(9 subjects, 22 ch, 250 Hz — genuinely different lab/hardware). `paradigm.get_data(dataset, subjects,
return_epochs=False)` → `(X 3D, y_str, meta)`; map `{"left_hand":0, "right_hand":1}`. The existing
`Pipeline([MyCSP, LDA])` runs unchanged (MyCSP is channel-count-agnostic). Treated as a **separate,
independent run** — NOT merged with PhysioNet (different ch/sfreq → different feature space).

**Quarantine:** `import moabb` is **lazy (inside the function)** with a clear
`pip install -r requirements-bonus.txt` message if missing. `requirements-bonus.txt` lists `moabb>=1.5.0`
**plus its core deps pinned to the mandatory windows** (`mne>=1.10,<1.13`, `scikit-learn>=1.5,<1.7`,
`numpy>=1.26,<2.3`, `scipy>=1.13,<1.18`) so installing the bonus can never silently upgrade a
mandatory-pinned lib. Never touch `requirements.txt`. Verification runs in a throwaway `.venv-bonus`
so the main `.venv` stays pristine.

**Tests** (`tests/test_external.py`): guarded by `pytest.importorskip("moabb")` AND `@pytest.mark.network`.
Shape-compatibility proven by feeding synthetic `(n, 22, T)` through `build_pipeline` (no moabb needed);
the real `load_external(1)` smoke checks `X.ndim==3`, `set(y)=={0,1}`, and prints `cross_val_score` as an
**observation, never an assert** (per-subject 2a accuracy ~0.5–0.9 would flake any threshold).

---

## Safeguards (must-implement — from adversarial critique)

- **A / pickle invariant:** `solver` is read ONLY in `fit()`, never in `transform`/`predict`, so
  previously-persisted joblib artifacts in `models/` (which bypass `__init__` on unpickle and lack
  `solver`) stay loadable and predictable.
- **A / PD guard:** `generalized_eigh` raises a clear error if any whitening eigenvalue ≤ 0 (instead of
  silent NaNs) — guards against a future `reg=0`.
- **A / existing test:** update `tests/test_csp.py` `get_params()` expectation to include
  `"solver": "eigh"` **in the same commit** as the `csp.py` `__init__` change.
- **C / CLI safety:** `tune` is a 3-arg mode (no arity change); update `tests/test_cli.py`; keep the 6
  existing CLI tests green.
- **C / determinism:** inner `StratifiedKFold` uses a fixed `random_state=42` for a stable demo.
- **D / class mapping:** the `mu0↔classes_[0]`, `mu1↔classes_[1]` invariant is explicit and tested on
  asymmetric data (label agreement, not just accuracy).
- **D / run_all isolation:** `own-lda` is NOT added to `evaluate.CLASSIFIERS`.
- **F / epoch-length guard:** `FilterBankCSP.fit` raises if the largest FIR filter length ≥ `n_times`
  (current n_times=321 vs max FIR ≈265 is fine; must fail loudly if TMIN/TMAX/SFREQ ever shrink).
- **F / leakage:** band-pass strictly inside `fit`/`transform`; never pre-filter X outside the pipeline.
- **G / dependency pinning:** `requirements-bonus.txt` pins moabb's core transitive deps to the mandatory
  windows; bonus install must not perturb the verified 0.658 numerics.
- **G / comparison framing:** PhysioNet and BNCI runs are two **independent** evaluations, not a merged
  dataset (different channel count / sampling rate).

---

## Clean file plan

**New:** `tpv/jacobi.py`, `tpv/own_lda.py`, `tpv/fbcsp.py`, `tpv/external.py`, `requirements-bonus.txt`,
`tests/test_jacobi.py`, `tests/test_own_lda.py`, `tests/test_fbcsp.py`, `tests/test_external.py`,
`scripts/bonus_demo.py`.

**Edits (all additive, no default changed):** `tpv/csp.py` (`solver` kwarg + one if/else);
`tpv/pipeline.py` (`own-lda` + `fbcsp` branches); `tpv/evaluate.py` (`tune`); `tests/test_csp.py`
(get_params dict); `mybci.py` (`tune` mode only — no flags that change arity); `README.md` (Bonus section).

**Net new public surface:** 1 `MyCSP` kwarg (`solver`); 2 `build_pipeline` enum values
(`clf="own-lda"`, `csp="fbcsp"`); 1 `evaluate` function (`tune`); 1 module function (`load_external`).

**Rejected (to honor minimal complexity):** plugin/ABC framework; dataset-aware `build_dataset`;
`build_dataset_wide` broadband loader; MIBIF; FFT band-power (item E); Ledoit-Wolf in OwnLDA; nested
CV / large grids on the default path; moabb's evaluation framework.

## Build order

**A → C + D → F → G → bonus_demo.** Land + `pytest -m "not network"` after each so any regression is
caught immediately. The mandatory path's tests must stay green throughout.

## Demo

`scripts/bonus_demo.py` — one CLI demonstrating all five: jacobi parity, OwnLDA parity, FBCSP run,
tuning best-params, and (if moabb installed) the second dataset, on one subject. Doubles as living docs.
