# Total Perspective Vortex

EEG brain-computer interface (BCI) that infers which movement a subject performs or imagines
(e.g. left vs right hand) from PhysioNet EEG, using a **from-scratch CSP** dimensionality-reduction
transformer inside a scikit-learn pipeline.

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate          # afterwards just use `python` / `pytest`
pip install -r requirements.txt
```
The PhysioNet `eegmmidb` dataset (~3.1 GB) auto-downloads into the **project-local `mne_data/`**
folder on first run (gitignored — never committed). Run every command from the repository root.

## Evaluation walkthrough (run in this order)
```bash
source .venv/bin/activate

# 0) ONE-TIME (do before the defense): fetch the dataset (~3.1 GB) + confirm the 60% gate.
#    Slow on first run (download); ~2.5 min afterwards.
python scripts/validate_60.py

# 1) Tests — all pass, no warnings.
pytest -m "not network"            # fast unit tests (config, CSP, pipeline, CLI)
pytest                             # full suite (incl. CSP-vs-MNE parity, train/predict smoke)

# 2) Preprocessing: raw vs 7-30 Hz filtered signal + PSD (close the windows to continue).
python scripts/visualize.py 4 14

# 3) Train: prints the cross_val_score fold array + mean, then saves the model.
python mybci.py 4 14 train

# 4) Predict: streams the held-out epochs, prints prediction vs truth, < 2 s per chunk.
python mybci.py 4 14 predict

# 5) (optional) Prove the from-scratch CSP matches mne.decoding.CSP (|Δacc| < 0.05).
pytest tests/test_csp_parity.py -v

# 6) Score: full 109-subject x 6-experiment mean (must be >= 60%).
python scripts/validate_60.py --fast 5     # quick preview (~10 s)
python scripts/validate_60.py              # full gate (~2.5 min once cached)
python mybci.py                            # subject-spec form (prints the 6 means + grand mean)
```

## Usage details
- `python mybci.py <subject 1-109> <run 3-14> <train|predict>`
- `<run>` selects an experiment: `3/7/11`→exp0, `4/8/12`→exp1, `5/9/13`→exp2, `6/10/14`→exp3.
  (e.g. `4 14` = subject 4, run 14 → exp3 = imagery hands vs feet.)
- Experiments 4/5 (real vs imagined) are evaluated only by `python mybci.py` / `validate_60.py`.

## The 6 experiments
| Exp | Runs | Task |
|-----|------|------|
| 0 | 3,7,11 | execution L vs R hand |
| 1 | 4,8,12 | imagery L vs R hand |
| 2 | 5,9,13 | execution hands vs feet |
| 3 | 6,10,14 | imagery hands vs feet |
| 4 | 3,7,11 vs 4,8,12 | execution vs imagery (hands) |
| 5 | 5,9,13 vs 6,10,14 | execution vs imagery (bilateral) |

## How it works
1. **Load / preprocess** (`tpv/data.py`, `tpv/preprocessing.py`): load EDF via MNE, average
   reference, 7-30 Hz band-pass (mu + beta sensorimotor rhythms), epoch with run-aware T1/T2 labels.
2. **CSP** (`tpv/csp.py`): from-scratch Common Spatial Patterns — per-class covariance +
   generalized eigendecomposition (`scipy.linalg.eigh`), normalized log-variance features.
   A `BaseEstimator` + `TransformerMixin`, validated against `mne.decoding.CSP`.
3. **Pipeline** (`tpv/pipeline.py`): `Pipeline([CSP, LDA])` — CSP is the first step so it is
   refit on each CV training fold (no data leakage).
4. **Validate** (`tpv/evaluate.py`): `cross_val_score` per subject; the mean of the 6
   experiment means must be >= 60% (achieved ~0.658).
5. **Stream** (`tpv/predict.py`): replays held-out epochs as a simulated stream, < 2 s/chunk
   (mne-realtime is not used).

## Data & artifacts (gitignored — only the code is committed)
- `mne_data/` — the PhysioNet dataset (~3.1 GB), project-local so it travels with the folder.
- `models/` — trained pipelines saved by `train`, loaded by `predict` (regenerable any time).

## Reproducibility
`TPV_SEED=42 python mybci.py 4 14 train` for a deterministic run; unset = different splits each
time (per the subject's "different splits each time"). The >= 60% bar applies to the full-run
grand mean, not to a single `train` call.

## Bonus (all opt-in — the mandatory path above is unchanged)

Five extra features, none of which alter the default `train` / `predict` / `run_all` behavior:

| Bonus | How to use | Eval box |
|---|---|---|
| **A. From-scratch eigensolver** | `MyCSP(solver="jacobi")` — pure-numpy cyclic Jacobi + whitening (matches scipy `eigh` to ~1e-14) | Implementations |
| **C. Hyperparameter tuning** | `python mybci.py <subject> <run> tune` — leakage-free GridSearchCV over `n_components` | Implementations |
| **D. From-scratch classifier** | `build_pipeline(clf="own-lda")` — numpy `OwnLDA` | Implementations |
| **F. Filter-Bank CSP** | `build_pipeline(csp="fbcsp")` — 4 sub-band CSP features | Feature engineering |
| **G. Second dataset** | `tpv.external.load_external()` — BCI Competition IV-2a via moabb (optional dep) | Datasets |

Demo all five on one subject:
```bash
python scripts/bonus_demo.py        # subject 1 (runs all five, including the 2nd dataset)
```

Everything — mandatory + all five bonuses — runs in the **single venv** from one
`pip install -r requirements.txt` (`moabb`, used by item G, is included). Installing moabb does not
change the pinned core libraries (numpy / scipy / scikit-learn / mne), so the mandatory numerics
(0.658) are unaffected.
