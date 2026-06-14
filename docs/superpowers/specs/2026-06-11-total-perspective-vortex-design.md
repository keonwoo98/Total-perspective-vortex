# Total Perspective Vortex — Design Spec (final)

> EEG Brain-Computer Interface (BCI) on the PhysioNet EEG Motor Movement/Imagery dataset.
> A **from-scratch CSP** dimensionality-reduction transformer, an anti-leakage scikit-learn `Pipeline`,
> simulated-stream prediction under a 2 s budget, and a **≥ 60 % mean accuracy** target across
> 6 experiments × 109 subjects.
>
> **Scope (locked by user):** mandatory + core bonus — from-scratch CSP (covariance + generalized
> eigendecomposition, validated against `mne.decoding.CSP`), multi-classifier comparison, and the full
> 109-subjects × 6-experiments evaluation. **Not** full bonus: no wavelets, no from-scratch SVD,
> no extra datasets (those are deferred — see §13).
>
> Date: 2026-06-11. All facts verified against physionet.org, mne.tools, scikit-learn.org.

---

## 1. Overview & goals

Build `mybci.py`, a single CLI program that:
1. loads PhysioNet EEG motor-imagery/execution recordings with MNE,
2. preprocesses (average reference, 7–30 Hz band-pass, run-aware epoching),
3. reduces dimensionality with a **custom CSP transformer** (`BaseEstimator + TransformerMixin`, numpy/scipy only),
4. classifies inside a scikit-learn `Pipeline` evaluated with `cross_val_score`,
5. replays a held-out split as a **simulated data stream**, emitting one prediction per chunk within a **2 second** wall-clock budget (mne-realtime forbidden).

**Hard requirements (PDF p.6–8):**
- Custom dimensionality reduction as a sklearn transformer using `BaseEstimator` + `TransformerMixin`.
- Use the sklearn `Pipeline`; run `cross_val_score` on the **whole** pipeline.
- Predict on a streamed/"playback" chunk and emit output **before a 2 s delay** after the chunk is handed to the pipeline. **No mne-realtime.**
- Achieve **≥ 60 % mean accuracy** across the six experiment types on never-learned data, averaged over all 109 subjects (PDF example grand mean = `0.6261`).

**Success criteria:**
1. `python mybci.py <subject> <run> train` prints the per-fold `cross_val_score` array + its mean.
2. `python mybci.py <subject> <run> predict` streams held-out epochs, printing `epoch nb: [pred] [truth] equal?` per chunk + a final `Accuracy:`, each prediction under 2 s latency.
3. `python mybci.py` (no args) runs all 6 experiments × 109 subjects and prints per-experiment means + a grand mean `≥ 0.60`.
4. The from-scratch CSP reproduces `mne.decoding.CSP` cross-val accuracy within tolerance (|Δacc| < 0.05) on the same epochs.

**Margin caveat (load-bearing):** 60 % is a **thin** bar. The published CSP+LDA cross-subject baseline on this dataset is ≈ 60 %; the PDF's own grand mean is 0.6261 with imagery exp1 the weakest (≈ 0.57). The ≈ 93 % single-subject figure from the MNE tutorial is **not** representative of the across-subjects mean — do not cite it as comfort. What keeps the mean above 60 % is (a) correct per-run-group T1/T2 label building and (b) zero data leakage.

---

## 2. Dataset & the 6-experiment run mapping

**Dataset:** PhysioNet "EEG Motor Movement/Imagery Database" (`eegmmidb` v1.0.0), recorded with BCI2000.
- 109 subjects (S001–S109), 64 EEG channels (10–10 system), **160 Hz**, EDF+ with TAL annotation channels.
- 14 runs/subject: Run 1 = baseline eyes-open, Run 2 = baseline eyes-closed (both **unused**); runs 3–14 = twelve 2-minute task runs (3 repetitions of 4 task conditions).

**Authoritative run-to-task table (MNE `eegbci.load_data` docs):**

| Runs | Task | T1 means | T2 means |
|------|------|----------|----------|
| 1 | Baseline, eyes open | — | — |
| 2 | Baseline, eyes closed | — | — |
| 3, 7, 11 | Motor **execution**: left vs right hand (fist) | left fist | right fist |
| 4, 8, 12 | Motor **imagery**: left vs right hand (fist) | left fist | right fist |
| 5, 9, 13 | Motor **execution**: hands vs feet | both fists | both feet |
| 6, 10, 14 | Motor **imagery**: hands vs feet | both fists | both feet |

**Annotation semantics (CRITICAL — highest-impact correctness risk):**
- `T0` = rest in **all** runs → **dropped** for binary tasks.
- `T1`/`T2` meanings are **run-group-dependent**. The EDF codes alone do NOT encode this; labels MUST be assigned conditioned on the run number, or left-fist gets silently merged with both-fists and accuracy collapses to chance.

**The six experiments** (exp0–3 fixed by the dataset; exp4/exp5 a documented design choice, **confirmed** as the real-vs-imagined contrast):

| Exp | Runs (class 0) | Runs (class 1) | Task | Class 0 | Class 1 |
|-----|------|------|------|---------|---------|
| 0 | 3,7,11 (T1) | 3,7,11 (T2) | Execution, L vs R hand | left fist | right fist |
| 1 | 4,8,12 (T1) | 4,8,12 (T2) | Imagery, L vs R hand | left fist | right fist |
| 2 | 5,9,13 (T1) | 5,9,13 (T2) | Execution, hands vs feet | both fists | both feet |
| 3 | 6,10,14 (T1) | 6,10,14 (T2) | Imagery, hands vs feet | both fists | both feet |
| 4 | 3,7,11 (T1+T2) | 4,8,12 (T1+T2) | **Execution vs imagery**, hands | real fist movement | imagined fist movement |
| 5 | 5,9,13 (T1+T2) | 6,10,14 (T1+T2) | **Execution vs imagery**, bilateral | real bilateral movement | imagined bilateral movement |

**Label-construction recipe (must be byte-identical across train/predict/run_all):**
- exp0–3: within one run-group, class 0 = T1 epochs, class 1 = T2 epochs. Drop T0.
- exp4/5: pool **both** motion epochs (T1 and T2) from each run-group; class is the **group provenance** (real vs imagined). Drop T0. (Within-class covariance therefore mixes left/right or fists/feet — this is intentional for a real-vs-imagined contrast and disclosed at defense.)
- The mapping lives in **one editable dict** in `tpv/config.py`.

**Single-run addressability (resolves a run→experiment ambiguity):** each run 3–14 belongs to exactly one of exp0–exp3 (3/7/11→exp0, 4/8/12→exp1, 5/9/13→exp2, 6/10/14→exp3), so the CLI single-run forms (`train`/`predict <subject> <run>`) map a run **uniquely to exp0–exp3**. exp4/exp5 are **composite** experiments (they span two run-groups, e.g. runs 3/7/11 appear in both exp0 and exp4), so they are **not single-run addressable** and are evaluated **only in `run_all`**.

**Problematic subjects:** S088, S089, S092, S100 have known run-length / sample-rate anomalies in some copies. **Policy:** at load time guard `sfreq == 160.0` and `len(eeg_picks) == 64`; on a sample-rate mismatch call `raw.resample(160.0)` and **keep the subject** so the denominator stays 109. If a specific cell still cannot be epoched, log it and average over the successfully-evaluated subjects, **printing the actual N** for that experiment.

---

## 3. Preprocessing & filtering parameters

Per (subject, experiment) — i.e. per run-group:

1. **Load:** `mne.datasets.eegbci.load_data(subject, runs, path=None)` — pass subject/runs **positionally** (the first parameter was renamed `subject`→`subjects` in MNE 1.9, so positional works across the pinned 1.6–1.12 range; a `subjects=` keyword would raise `TypeError` on < 1.9). `path`/`update_path`/etc. are **keyword-only**. Auto-downloads EDF to `~/mne_data`. Read each file with `mne.io.read_raw_edf(fname, preload=True)`.
2. **Standardize channel names:** `mne.datasets.eegbci.standardize(raw)` (in-place; strips trailing dots like `Fc5.`, `Cz..`). Must precede montage.
3. **Concatenate** the 3 runs of the group: `mne.concatenate_raws([...])` (inserts `edge` annotations at joins). For exp4/5, concatenate all 6 runs but track provenance for labels.
4. **Montage:** `raw.set_montage(mne.channels.make_standard_montage('standard_1005'))`. Needed for plotting/topomaps only, not for CSP accuracy.
5. **Reference:** average reference, `raw.set_eeg_reference(ref_channels='average')`. (Reduces spatial-covariance rank to `n_channels − 1`, so CSP must regularize — see §4.)
6. **Band-pass the CONTINUOUS raw (before epoching):**
   `raw.filter(7.0, 30.0, fir_design='firwin', skip_by_annotation='edge')`
   - **7–30 Hz** = mu (8–12) + beta (13–30) sensorimotor rhythms.
   - zero-phase FIR (firwin/Hamming): no temporal distortion for time-locked epochs.
   - `skip_by_annotation='edge'` filters each contiguous segment independently (no ringing across concat joins).
   - **No 50/60 Hz notch** — the 30 Hz high-cut already removes line noise; state this at defense.
7. **Events:** `events, _ = mne.events_from_annotations(raw, event_id=...)`. Build `event_id` per run-group so T1/T2 carry the correct class; select only the motion annotations (exclude T0).
8. **Epoch:** `mne.Epochs(raw, events, event_id, tmin=0.0, tmax=2.0, baseline=None, picks='eeg', preload=True)`.
   - **Window `tmin=0.0, tmax=2.0`** (confirmed): a ≤ 2 s window honoring the real-time constraint, identical for train and predict.
   - `baseline=None` is required for CSP (covariance-based; baseline subtraction distorts covariance).
9. **Feature array:** `X = epochs.get_data(copy=True)` → `(n_epochs, 64, n_times)`; `y = epochs.events[:, -1]` remapped to `{0, 1}`.

**Channels:** all 64; CSP does the spatial selection (matches MNE example).

**Visualizations for defense:** `raw.plot()` before filter, after 7–30 Hz; `raw.compute_psd(method='welch').plot()` before/after (justify the band; `psd_welch` is deprecated — use `compute_psd`); optionally `csp.plot_patterns(epochs.info)` to show C3/Cz/C4 sensorimotor focus.

---

## 4. From-scratch CSP transformer

### 4.1 What it is + subject-formalism reconciliation (load-bearing)

CSP is a **supervised binary spatial filter** operating on **channel × channel** spatial covariance (computed across time within each epoch) — **not** on a flattened `ch*time` vector.

The PDF writes `X ∈ ℝ^(d×N)` with `d = ch*time` and asks for `W` s.t. `Wᵀ X = X_CSP`. Reconciliation to state at defense:

> "Each column of `X` is one trial; `W` is the **spatial-filter matrix** (`n_components × n_channels`) applied within each trial as `Wᵀ Eₙ`. The rows of `X_CSP` are the CSP components. We never build a `(ch*time)×(ch*time)` matrix — it would be rank-deficient and match no real CSP. CSP is the spatial-covariance special case of the projection; when class covariances are equal it reduces to PCA."

### 4.2 Math

For class covariances `C1`, `C2` (each `n_channels × n_channels`), solve the **generalized eigenvalue problem**:

```
C1 w = λ (C1 + C2) w     ⇔     scipy.linalg.eigh(C1, C1 + C2)
```

Eigenvalues lie in `[0, 1]`; `λ → 1` = high class-1 variance, `λ → 0` = high class-2 variance. Discriminative filters have `λ` farthest from 0.5.

### 4.3 numpy/scipy `fit(X, y)` recipe

1. Split epochs by label into class-0 / class-1.
2. Per epoch `E` (`n_channels × n_times`): `C = E @ E.T`; trace-normalize `C /= np.trace(C)`. Average within each class → `C1`, `C2`. (Ramoser per-trial `cov_est='epoch'` route — **fixed** as the project's route; the parity test in §4.6 configures MNE to match.)
3. Symmetrize: `C = (C + C.T) / 2`.
4. **Regularize (required after average reference):** `C = (1−reg)*C + reg*(np.trace(C)/n_ch)*np.eye(n_ch)`, `reg = 0.01`, so `C1+C2` is positive-definite (`scipy.linalg.eigh` requires its `b` operand PD or it raises `LinAlgError`).
5. Solve: `eigvals, eigvecs = scipy.linalg.eigh(C1, C1 + C2)` — use **`scipy.linalg.eigh`**, never `numpy.linalg.eig` (eigh → real eigenvalues + b-orthonormal vectors).
6. Order by descending `|λ − 0.5|`; keep the first `n_components` (= 4).
7. `self.filters_ = eigvecs[:, order].T[:n_components]`; optionally `self.patterns_ = scipy.linalg.pinv(eigvecs.T)`.
8. `return self`.

### 4.4 `transform(X)` recipe

Per epoch `E`: `Z = self.filters_ @ E`; per-component feature `f_i = log( var(Z_i) / Σ_j var(Z_j) )` (normalized log-variance, `np.var(Z, axis=1)`). Return `(n_epochs, n_components)`.

> MNE's default feature is the **unnormalized** `np.log((Z**2).mean(axis=2))`. The normalized form differs by a **per-epoch** (not global) additive term inside the log, so it is not a strict affine map; empirically the CV-accuracy difference stays well under the 0.05 parity tolerance. The normalized form matches the PDF text.

### 4.5 sklearn interface

- `class MyCSP(BaseEstimator, TransformerMixin)`.
- `__init__(self, n_components=4, reg=0.01)`: **store hyperparameters only** — no validation, no renaming (so `clone`/`set_params`/CV-refit work).
- `fit(self, X, y)`: compute and store `self.filters_` (trailing underscore = learned state, reset each fit); `return self`.
- `transform(self, X)`: 3D `(n_epochs, n_channels, n_times)` → 2D `(n_epochs, n_components)`; must not change sample count/order.
- Accept `y=None` default so a `Pipeline` can pass `y` positionally.
- `n_components` is **fixed at 4 globally** (no per-experiment tuning → no selection leakage). `reg=0.01` default.

### 4.6 Validation-against-MNE plan (quantified)

1. Build the MNE reference: 7–30 Hz, epochs `tmin=0,tmax=2`, `mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=True, cov_est='epoch')` + `LDA`, `ShuffleSplit(10, test_size=0.2, random_state=42)`, `cross_val_score` — on a fixed cell (subject 1, exp3 = runs 6/10/14).
2. Swap in `MyCSP(n_components=4)`; assert **`|mean(cv_scratch) − mean(cv_mne)| < 0.05`** on the same epochs with the same `random_state=42`.
3. Compare `filters_` only up to **sign/scaling** (eigenvector sign is arbitrary; squared log-variance is sign-invariant — not a bug).
4. Self-checks: eigenvalues ∈ `[0, 1]`; on failure, investigate covariance route / `reg`.
5. **No from-scratch SVD** — `scipy.linalg.eigh` is the from-scratch boundary for the core bonus.

---

## 5. sklearn Pipeline & anti-leakage design

```python
clf = Pipeline([
    ('csp', MyCSP(n_components=4, reg=0.01)),
    ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
])
```

**Anti-leakage (the single most load-bearing methodology requirement):** CSP is supervised (uses `y`). It MUST be a `Pipeline` step fit **inside** `cross_val_score`, so each fold's CSP is fit on the **training fold only**. Fitting CSP on the full dataset before CV leaks test-class covariance into the spatial filters. Passing the whole `Pipeline` to `cross_val_score` enforces correct per-fold `fit`/`predict` automatically.

**Classifier comparison:** `pipeline.py` can swap the classifier; `evaluate` compares LDA / linear-SVM / LogisticRegression / RandomForest via `cross_val_score` and the **default is the best (expected: `LDA(solver='lsqr', shrinkage='auto')`)**. Note `shrinkage` works only with `lsqr`/`eigen` solvers (the default `svd` silently ignores it). Pass `config.get_seed()` to any `random_state`-bearing estimator (RF/SVM).

---

## 6. train / validation / test strategy

**The CLI `run` selects the experiment group** (the 3 runs sharing the task; 6 runs for exp4/5). `mybci.py 4 14 train` → subject 4, run 14 ∈ exp3 (runs 6/10/14), concatenated → ≈ 45 epochs. This reconciles with the PDF example: 45 epochs × `ShuffleSplit(test_size=0.2)` ⇒ 9-sample test folds ⇒ scores are k/9 (PDF `[6/9,4/9,…] = 0.5333`), and the predict held-out = 0.2 × 45 ≈ 9 epochs (PDF epoch 00–08).

Per-subject, **within-subject** (CSP filters are subject-specific; "never-learned data" = held-out epochs of the **same** subject):

1. `scores = cross_val_score(clf, X, y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=seed))`.
   - Returns one float per fold; print the array then `cross_val_score: {scores.mean():.4f}`.
   - `cv` is set **explicitly** to `ShuffleSplit(10, …)` (PDF train example shows 10 numbers).
2. Deployable model for `predict`: `X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)`; `clf.fit(X_tr, y_tr)`.
3. Persist **one artifact** = `{ 'pipeline': clf, 'X_test': X_te, 'y_test': y_te, 'meta': {subject, experiment, window, n_components, …} }` via `joblib.dump`. **`predict` replays exactly these `X_test` epochs** → no seed-fragility, no re-split mismatch.

> **The ≥ 60 % gate applies ONLY to the `run_all` 6-experiment grand mean** — never to an individual `train` call. A single `train` can legitimately score below 60 % (the PDF's own example = 0.5333). State this explicitly so a 0.53 train output is not read as failure.

---

## 7. Streaming / playback predict design (< 2 s)

The "stream" is **simulated playback** of the persisted held-out epochs in chronological (event-onset) order. mne-realtime forbidden.

**Latency strategy (2 s budget trivially met):**
- All heavy work **before** the loop: `joblib.load(artifact)` once → `pipeline`, `X_test`, `y_test`.
- Inside the loop: slice one epoch `X_test[i:i+1]` (`(1, 64, n_times)`), call `pipeline.predict(...)` → length-1 array. CSP transform = a few matmuls + log-variance; LDA predict = one dot product → sub-millisecond.
- Wrap predict in `time.perf_counter()` and assert `< 2.0 s`.
- An optional cosmetic `time.sleep` to emulate cadence must NOT count against measured latency. **Do not parallelize the predict loop.**

**Required output (PDF p.8 format):**
```
epoch nb: [prediction] [truth] equal?
epoch 00:    [2]    [1] False
...
epoch 08:    [2]    [2] True
Accuracy: 0.6666
```
(Class labels printed as the original event ints `1`/`2` for parity with the PDF, mapped back from internal `{0,1}`.)

---

## 8. The three CLI forms

| Command | Meaning | Output |
|---------|---------|--------|
| `python mybci.py <subject> <run> train` | Train on (subject, run→experiment group) | per-fold `cross_val_score` array + `cross_val_score: <mean>`, then persist artifact |
| `python mybci.py <subject> <run> predict` | Replay persisted held-out epochs as a stream | `epoch nb: [pred] [truth] equal?` per chunk + `Accuracy: <float>` |
| `python mybci.py` (no args) | All 6 experiments × 109 subjects | per-subject accuracy, per-experiment mean, then grand mean (`≥ 0.60`) |

**Arg semantics (documented inference, not verbatim PDF):** `4 14` = subject 4, run 14. `run ∈ 3..14` is the literal PhysioNet run, mapped **uniquely** to one of exp0–exp3 (3/7/11→exp0, 4/8/12→exp1, 5/9/13→exp2, 6/10/14→exp3). Runs in the same group are equivalent (e.g. `train 4 3` ≡ `train 4 7`). exp4/exp5 are composite and addressed **only by `run_all`**, never by a single run. **Document this at defense.**

**argparse:** positionals `subject` (int), `run` (int), `mode` (`choices=['train','predict']`), all `nargs='?'`. Dispatch: no args → `run_all()`; `subject`+`run`+`mode` present → dispatch on mode; otherwise print usage + exit non-zero. Validate `subject ∈ 1..109`, `run ∈ 3..14`. On `predict` with a missing artifact → clear error telling the user to `train` first. `run_all` averages per-experiment first, then the 6 experiment means.

---

## 9. Project file structure & module responsibilities

Git turn-in = **code only** (no dataset, no models).

```
Total-perspective-vortex/
  mybci.py                 # thin CLI dispatcher (argparse → train/predict/run_all)
  requirements.txt
  README.md
  .gitignore
  tpv/
    __init__.py
    config.py              # SFREQ=160, FMIN/FMAX=7/30, TMIN=0/TMAX=2, N_COMPONENTS=4, REG=0.01,
                           #   EXPERIMENTS run-map dict, MODELS_DIR, get_seed(); stdlib only
    data.py                # load_raw(subject, runs) → Raw (load_data + read_raw_edf + standardize + concat + resample-guard)
    preprocessing.py       # filter_raw(raw); make_epochs(raw, experiment) → (X, y) with run-aware T1/T2 labels
    csp.py                 # MyCSP(BaseEstimator, TransformerMixin): numpy cov + scipy.linalg.eigh
    pipeline.py            # build_pipeline(csp='scratch'|'mne', clf='lda'|'svm'|'logreg'|'rf') → Pipeline
    train.py               # train(subject, run): cross_val_score print, fit, joblib.dump artifact
    predict.py             # predict(subject, run): joblib.load, stream loop, per-chunk print, accuracy
    evaluate.py            # compare_classifiers(...); run_all() over 109×6 with the 60% gate
    viz.py                 # optional raw/PSD/CSP-pattern plots; lazy matplotlib import
  tests/
    test_csp_parity.py     # MyCSP vs mne.decoding.CSP cross-val accuracy, |Δ| < 0.05
    test_pipeline_smoke.py # one subject trains, scores > chance; artifact round-trips
  scripts/
    validate_60.py         # run_all → exit 1 if grand mean < 0.60 (pre-defense gate)
```

Single responsibility per module: `config` = pure constants/no I/O; `data` = EDF fetch/read/standardize/concat/guard; `preprocessing` = filter + run-aware epoching; `csp` = the transformer; `pipeline` = factory; `train`/`predict`/`evaluate` = the three behaviors; `viz` = optional plots (lazy import so headless runs never crash).

---

## 10. Build / verification sequence (library-first → from-scratch)

1. **Env:** `python3.12 -m venv .venv` (3.12.1 installed; 3.10.2 too old, 3.14.3 too new). Install requirements; confirm `import mne, sklearn, scipy, numpy`.
2. **data + preprocessing:** download subject 1, build epochs; assert `X.shape == (n_epochs, 64, n_times)`, `y ∈ {0,1}`, `sfreq == 160`.
3. **Architecture with `mne.decoding.CSP` + LDA:** `evaluate` with `ShuffleSplit`/`cross_val_score` on subject 1 → expect ≈ 0.8–0.93, confirm `≥ 0.60`. **Architecture validated.**
4. **train/predict** around the working mne-CSP pipeline; verify the joblib artifact round-trip and the streaming print format.
5. **From-scratch `csp.py`;** `test_csp_parity.py` asserts `|Δacc| < 0.05` vs mne on the fixed cell. Swap default to scratch CSP.
6. **Multi-classifier comparison** in `pipeline.py`/`evaluate.py`; `run_all()` over 109×6; `scripts/validate_60.py` gates mean `≥ 0.60`.
7. **viz + README** last.

---

## 11. Validating the ≥ 60 % mean accuracy

`scripts/validate_60.py` calls `evaluate.run_all()`:
- Loop 6 experiments × 109 subjects; per cell run `cross_val_score(pipeline, X, y, cv=ShuffleSplit(10, 0.2)).mean()`; collect.
- Print a per-experiment mean table + grand mean; `sys.exit(1)` if grand mean `< 0.60`.
- **Fast mode** (subject subset) for iteration; **full mode** for the pre-defense run.

**Performance/scale:** `run_all` ≈ 6 × 109 cells, dominated by MNE load+filter+epoch per subject (seconds each) → tens of minutes single-threaded. **Cache** filtered epochs to disk keyed by `(subject, runs)` to skip re-IO. `joblib.Parallel(n_jobs=...)` across **subjects** is fine; do NOT parallelize the predict stream.

**Margin management:** report the mean over `ShuffleSplit` folds; offer a deterministic `TPV_SEED=42` run as a reproducible demo. `n_components` is fixed at 4 (no per-experiment tuning → no selection leakage). Guard/resample S088/S089/S092/S100.

---

## 12. Requirements & reproducibility

**Interpreter (verified locally):** Python **3.10.2**, **3.12.1**, **3.14.3** are installed; **3.11 and 3.13 are absent**; **no** numpy/scipy/sklearn/mne installed on any of them. **Use Python 3.12** in a dedicated venv (mature cp312 wheels; 3.10.2 is an old patch, 3.14.3 risks old-pin source builds).

**`requirements.txt` (pin-and-freeze after one verified install):**
```
mne>=1.6,<1.13
scikit-learn>=1.5,<1.7
numpy>=1.26,<2.3
scipy>=1.13,<1.18
matplotlib>=3.9
joblib>=1.3
```

**Reproducibility policy:** single source of truth `config.get_seed()` → `int(TPV_SEED)` if the env var is set, else `None`. Threaded into `ShuffleSplit`/`train_test_split` and `random_state`-bearing estimators. **Default (unset) = different splits each time** (honors PDF p.8 "different splits each time"); `TPV_SEED=42` = byte-stable demo. The 60 % gate is a mean over 10 folds × many subjects, robust to split randomness. Never call global `np.random.seed()`. Pin sklearn because joblib models are not portable across versions (and `joblib.load` runs arbitrary code — load only your own artifacts).

**`.gitignore`:** `.venv/`, `__pycache__/`, `*.pyc`, `mne_data/`, `models/`, `*.joblib`, `*.pkl`, `.DS_Store`.

**Model artifacts:** `models/subj{NN}_run{RR}_exp{E}.joblib` (gitignored); one `joblib.dump` of the artifact dict (pipeline + held-out test). `config.MODELS_DIR` centralizes the path; `train` does `mkdir(parents=True, exist_ok=True)`.

---

## 13. Deferred to full bonus (explicitly out of scope now)

- Wavelet / time-frequency decompositions (CSSP, FBCSP, Morlet).
- **From-scratch SVD** (this build hand-writes covariance + generalized eigendecomposition via `scipy.linalg.eigh` only).
- Additional datasets beyond PhysioNet `eegmmidb`.
- Hand-written Jacobi/QR eigenvalue routines.
- Cross-subject transfer learning / subject-independent models (CSP filters are subject-specific by design).
- A live-device real-time stream (satisfied by simulated playback; mne-realtime stays forbidden).

> Per the user's instruction, full bonus is recommended **only after** the mandatory + core-bonus build is complete, verified (≥ 60 % grand mean, CSP parity passing), and fully understood.

---

## Open items confirmed with user
- exp4/exp5 = **real-vs-imagined** contrast — **confirmed**.
- Epoch window `tmin=0.0, tmax=2.0` — **confirmed**.
