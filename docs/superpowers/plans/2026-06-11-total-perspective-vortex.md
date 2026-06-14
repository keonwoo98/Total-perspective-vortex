# Total Perspective Vortex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `mybci.py`, an EEG brain-computer interface that classifies PhysioNet motor-imagery/execution trials using a from-scratch CSP transformer inside a scikit-learn `Pipeline`, validated with `cross_val_score`, reaching ≥ 60 % mean accuracy across 6 experiments × 109 subjects, with simulated-stream prediction under a 2 s budget.

**Architecture:** A thin `mybci.py` CLI dispatches to a `tpv/` package: `data` (load EDF), `preprocessing` (filter + run-aware epoching → dataset), `csp` (custom `BaseEstimator`+`TransformerMixin` spatial filter via numpy covariance + `scipy.linalg.eigh`), `pipeline` (factory), `train`/`predict`/`evaluate` (the three CLI behaviors). CSP lives inside the `Pipeline` so it is refit per CV fold (no leakage). Build is library-first (validate with `mne.decoding.CSP`), then swap in the from-scratch CSP and assert parity.

**Tech Stack:** Python 3.12, MNE-Python, scikit-learn, numpy, scipy, joblib, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-total-perspective-vortex-design.md` (read it before starting).

---

## Locked interface contract (used consistently by all tasks)

These names/signatures are fixed. Later tasks depend on them exactly as written.

```
tpv/config.py
  SFREQ = 160.0
  FMIN, FMAX = 7.0, 30.0
  TMIN, TMAX = 0.0, 2.0
  N_COMPONENTS = 4
  REG = 0.01
  N_CHANNELS = 64
  N_SUBJECTS = 109
  MODELS_DIR : pathlib.Path = Path("models")
  EXPERIMENTS : dict[int, list[dict]]   # each class-spec: {"runs": [...], "annotations": [...], "label": 0|1}
  RUN_TO_EXPERIMENT : dict[int, int]    # run 3..14 -> exp 0..3
  runs_for_experiment(exp: int) -> list[int]
  get_seed() -> int | None

tpv/data.py
  load_raw(subject: int, runs: list[int]) -> mne.io.BaseRaw

tpv/preprocessing.py
  filter_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw
  epochs_from_raw(raw, annotations: list[str], label: int) -> tuple[np.ndarray, np.ndarray]
  build_dataset(subject: int, experiment: int) -> tuple[np.ndarray, np.ndarray]   # X (n,64,n_times) float64, y (n,) int {0,1}

tpv/csp.py
  class MyCSP(BaseEstimator, TransformerMixin):
    __init__(self, n_components: int = 4, reg: float = 0.01)
    fit(self, X, y=None) -> "MyCSP"   # sets self.filters_, self.patterns_, self.classes_, self.eigenvalues_
    transform(self, X) -> np.ndarray  # (n_epochs, n_components)

tpv/pipeline.py
  build_pipeline(csp: str = "scratch", clf: str = "lda",
                 n_components: int = 4, reg: float = 0.01, seed: int | None = None) -> sklearn.pipeline.Pipeline

tpv/train.py
  artifact_path(subject: int, run: int) -> pathlib.Path
  train(subject: int, run: int) -> float    # prints cv array+mean, fits, saves artifact, returns cv mean

tpv/predict.py
  predict(subject: int, run: int) -> float   # streams held-out epochs, prints per-chunk, returns accuracy

tpv/evaluate.py
  cross_val_accuracy(subject: int, experiment: int, csp="scratch", clf="lda") -> float
  compare_classifiers(subject: int, experiment: int) -> dict[str, float]
  run_all(subjects=range(1, 110), experiments=range(6), fast=False) -> float   # prints table, returns grand mean

mybci.py
  main(argv: list[str] | None = None) -> int
```

**Artifact format** (`joblib.dump`): a dict
`{"pipeline": fitted_Pipeline, "X_test": np.ndarray, "y_test": np.ndarray, "meta": {"subject", "run", "experiment", "tmin", "tmax", "n_components"}}`.

**Test markers:** tests that download EEG data are marked `@pytest.mark.network` and use subject 1 (cached after first run). Pure-logic tests (config, csp) use synthetic data and are fast/deterministic.

---

## File structure

```
Total-perspective-vortex/
  mybci.py
  requirements.txt
  README.md
  .gitignore
  pytest.ini
  tpv/
    __init__.py
    config.py
    data.py
    preprocessing.py
    csp.py
    pipeline.py
    train.py
    predict.py
    evaluate.py
    viz.py
  tests/
    __init__.py
    test_config.py
    test_csp.py
    test_preprocessing.py        # network
    test_csp_parity.py           # network
    test_pipeline_smoke.py       # network
    test_cli.py
  scripts/
    validate_60.py
```

---

## Task 0: Environment & scaffolding

**Files:**
- Create: `.venv/` (via venv), `requirements.txt`, `.gitignore`, `pytest.ini`, `tpv/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create the venv on Python 3.12**

Run:
```bash
/Users/keokim/.brew/bin/python3.12 -m venv .venv
.venv/bin/python --version
```
Expected: `Python 3.12.1`

- [ ] **Step 2: Write `requirements.txt`**

```
mne>=1.6,<1.13
scikit-learn>=1.5,<1.7
numpy>=1.26,<2.3
scipy>=1.13,<1.18
matplotlib>=3.9
joblib>=1.3
pytest>=8.0
```

- [ ] **Step 3: Install and verify imports**

Run:
```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import mne, sklearn, scipy, numpy, joblib, matplotlib; print(mne.__version__, sklearn.__version__, numpy.__version__, scipy.__version__)"
```
Expected: four version strings print with no ImportError.

- [ ] **Step 4: Freeze exact versions**

Run:
```bash
.venv/bin/pip freeze | grep -Ei "^(mne|scikit-learn|numpy|scipy|matplotlib|joblib)==" > requirements.lock.txt
```
Expected: `requirements.lock.txt` lists the resolved `==` pins. (Keep `requirements.txt` as ranges; `requirements.lock.txt` is the reproducible record.)

- [ ] **Step 5: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
mne_data/
models/
*.joblib
*.pkl
.DS_Store
.pytest_cache/
docs/superpowers/specs/*.bak
```

- [ ] **Step 6: Write `pytest.ini`**

```ini
[pytest]
markers =
    network: test downloads PhysioNet data (slow, needs internet)
testpaths = tests
```

- [ ] **Step 7: Create empty package markers**

```bash
touch tpv/__init__.py tests/__init__.py
```

- [ ] **Step 8: Commit**

```bash
git add requirements.txt requirements.lock.txt .gitignore pytest.ini tpv/__init__.py tests/__init__.py
git commit -m "chore: scaffold venv, deps, package skeleton"
```

---

## Task 1: `tpv/config.py` — constants, experiment map, seed

**Files:**
- Create: `tpv/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from tpv import config


def test_experiment_count_and_shape():
    assert set(config.EXPERIMENTS.keys()) == {0, 1, 2, 3, 4, 5}
    for exp, classes in config.EXPERIMENTS.items():
        assert len(classes) == 2, f"exp {exp} must have 2 classes"
        labels = sorted(c["label"] for c in classes)
        assert labels == [0, 1]
        for c in classes:
            assert set(c.keys()) == {"runs", "annotations", "label"}
            assert all(r in range(3, 15) for r in c["runs"])
            assert all(a in {"T1", "T2"} for a in c["annotations"])


def test_exp0_is_t1_vs_t2_same_runs():
    classes = config.EXPERIMENTS[0]
    assert classes[0]["runs"] == [3, 7, 11] and classes[0]["annotations"] == ["T1"]
    assert classes[1]["runs"] == [3, 7, 11] and classes[1]["annotations"] == ["T2"]


def test_exp4_is_real_vs_imagined_pooled():
    classes = config.EXPERIMENTS[4]
    assert classes[0]["runs"] == [3, 7, 11] and classes[0]["annotations"] == ["T1", "T2"]
    assert classes[1]["runs"] == [4, 8, 12] and classes[1]["annotations"] == ["T1", "T2"]


def test_run_to_experiment_unique_mapping():
    assert config.RUN_TO_EXPERIMENT[14] == 3
    assert config.RUN_TO_EXPERIMENT[3] == 0
    assert config.RUN_TO_EXPERIMENT[8] == 1
    assert config.RUN_TO_EXPERIMENT[9] == 2
    # every task run 3..14 maps to exactly one of exp0..3
    assert sorted(config.RUN_TO_EXPERIMENT.keys()) == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert set(config.RUN_TO_EXPERIMENT.values()) == {0, 1, 2, 3}


def test_runs_for_experiment():
    assert sorted(config.runs_for_experiment(0)) == [3, 7, 11]
    assert sorted(config.runs_for_experiment(4)) == [3, 4, 7, 8, 11, 12]


def test_get_seed_reads_env(monkeypatch):
    # get_seed() reads os.environ live, so no module reload is needed.
    monkeypatch.delenv("TPV_SEED", raising=False)
    assert config.get_seed() is None
    monkeypatch.setenv("TPV_SEED", "42")
    assert config.get_seed() == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tpv.config'`.

- [ ] **Step 3: Write `tpv/config.py`**

```python
"""Central constants, the 6-experiment run map, and seed policy. Stdlib only."""
import os
from pathlib import Path

SFREQ = 160.0
FMIN, FMAX = 7.0, 30.0
TMIN, TMAX = 0.0, 2.0
N_COMPONENTS = 4
REG = 0.01
N_CHANNELS = 64
N_SUBJECTS = 109
MODELS_DIR = Path("models")

# Each experiment is a list of two class-specs.
# A class-spec selects epochs by (runs, annotations) and assigns a binary label.
# exp0-3: T1 vs T2 within one run-group. exp4/5: real vs imagined, pooling T1+T2.
EXPERIMENTS = {
    0: [
        {"runs": [3, 7, 11], "annotations": ["T1"], "label": 0},
        {"runs": [3, 7, 11], "annotations": ["T2"], "label": 1},
    ],
    1: [
        {"runs": [4, 8, 12], "annotations": ["T1"], "label": 0},
        {"runs": [4, 8, 12], "annotations": ["T2"], "label": 1},
    ],
    2: [
        {"runs": [5, 9, 13], "annotations": ["T1"], "label": 0},
        {"runs": [5, 9, 13], "annotations": ["T2"], "label": 1},
    ],
    3: [
        {"runs": [6, 10, 14], "annotations": ["T1"], "label": 0},
        {"runs": [6, 10, 14], "annotations": ["T2"], "label": 1},
    ],
    4: [
        {"runs": [3, 7, 11], "annotations": ["T1", "T2"], "label": 0},
        {"runs": [4, 8, 12], "annotations": ["T1", "T2"], "label": 1},
    ],
    5: [
        {"runs": [5, 9, 13], "annotations": ["T1", "T2"], "label": 0},
        {"runs": [6, 10, 14], "annotations": ["T1", "T2"], "label": 1},
    ],
}

# A single run maps uniquely to one of exp0..3 (the four pure task groups).
RUN_TO_EXPERIMENT = {
    3: 0, 7: 0, 11: 0,
    4: 1, 8: 1, 12: 1,
    5: 2, 9: 2, 13: 2,
    6: 3, 10: 3, 14: 3,
}


def runs_for_experiment(exp: int) -> list[int]:
    """All distinct runs used by an experiment (across both class-specs)."""
    runs: list[int] = []
    for spec in EXPERIMENTS[exp]:
        for r in spec["runs"]:
            if r not in runs:
                runs.append(r)
    return runs


def get_seed() -> int | None:
    """Reproducibility policy: TPV_SEED env var -> int, else None (different splits each time)."""
    val = os.environ.get("TPV_SEED")
    return int(val) if val is not None else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add tpv/config.py tests/test_config.py
git commit -m "feat(config): constants, 6-experiment run map, seed policy"
```

---

## Task 2: `tpv/data.py` — load & assemble raw EDF

**Files:**
- Create: `tpv/data.py`
- Test: `tests/test_preprocessing.py` (shared network test file; this task adds the data test)

- [ ] **Step 1: Write the failing test**

`tests/test_preprocessing.py` (create with the data test first):
```python
import numpy as np
import pytest
from tpv import config, data


@pytest.mark.network
def test_load_raw_subject1_exp3_runs():
    raw = data.load_raw(1, [6, 10, 14])
    assert raw.info["sfreq"] == config.SFREQ
    assert len(raw.get_data(picks="eeg")) == config.N_CHANNELS
    # annotations T0/T1/T2 survived concatenation
    descs = set(raw.annotations.description)
    assert {"T0", "T1", "T2"}.issubset(descs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_preprocessing.py -v -m network`
Expected: FAIL with `ModuleNotFoundError: No module named 'tpv.data'`.

- [ ] **Step 3: Write `tpv/data.py`**

```python
"""Load PhysioNet eegmmidb EDF runs and assemble one Raw per run-group."""
import mne
from mne.datasets import eegbci

from tpv import config


def load_raw(subject: int, runs: list[int]) -> mne.io.BaseRaw:
    """Download (if needed), read, standardize, concatenate, montage, and
    sample-rate-guard the given runs for one subject.

    Returns a single concatenated Raw with EEG channels and T0/T1/T2 annotations.
    """
    # Pass subject/runs POSITIONALLY: the first param was renamed subject->subjects in
    # MNE 1.9, so positional works across the pinned 1.6-1.12 range (keyword would break <1.9).
    paths = eegbci.load_data(subject, runs, path=None, update_path=True)
    raws = []
    for p in paths:
        r = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
        eegbci.standardize(r)  # strip trailing dots: 'Fc5.' -> 'Fc5'
        raws.append(r)
    raw = mne.concatenate_raws(raws)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")

    # Sample-rate guard (S088/S089/S092/S100 anomalies): keep all 109 subjects.
    if raw.info["sfreq"] != config.SFREQ:
        raw.resample(config.SFREQ, verbose="ERROR")
    return raw
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_preprocessing.py::test_load_raw_subject1_exp3_runs -v`
Expected: PASS (first run downloads ~subject 1 runs to `~/mne_data`, then caches).

- [ ] **Step 5: Commit**

```bash
git add tpv/data.py tests/test_preprocessing.py
git commit -m "feat(data): load+standardize+concat EDF runs with sfreq guard"
```

---

## Task 3: `tpv/preprocessing.py` — filter + run-aware dataset

**Files:**
- Create: `tpv/preprocessing.py`
- Test: `tests/test_preprocessing.py` (add tests)

> Note: `build_dataset` + `epochs_from_raw` **supersede** the spec §9 `make_epochs(raw, experiment)` sketch. `build_dataset` owns the per-run-group load+filter so exp4/5 provenance labels (real vs imagined) are correct — a single shared `raw` cannot encode which run-group an epoch came from.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_preprocessing.py`:
```python
from tpv import preprocessing


@pytest.mark.network
def test_build_dataset_exp3_shapes_and_labels():
    X, y = preprocessing.build_dataset(1, 3)
    assert X.ndim == 3 and X.shape[1] == config.N_CHANNELS
    n_times_expected = int(round((config.TMAX - config.TMIN) * config.SFREQ)) + 1
    assert abs(X.shape[2] - n_times_expected) <= 1
    assert set(np.unique(y)).issubset({0, 1})
    assert len(np.unique(y)) == 2          # both classes present
    assert X.shape[0] == y.shape[0]
    assert X.dtype == np.float64


@pytest.mark.network
def test_build_dataset_exp4_pools_provenance():
    # exp4 = real (runs 3/7/11) label 0  vs  imagined (runs 4/8/12) label 1
    X, y = preprocessing.build_dataset(1, 4)
    assert set(np.unique(y)) == {0, 1}
    assert (y == 0).sum() > 0 and (y == 1).sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_preprocessing.py -v -k build_dataset`
Expected: FAIL with `AttributeError: module 'tpv.preprocessing' has no attribute 'build_dataset'`.

- [ ] **Step 3: Write `tpv/preprocessing.py`**

```python
"""Band-pass filtering and run-aware epoching into (X, y) datasets."""
import mne
import numpy as np

from tpv import config, data


def filter_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Average reference + 7-30 Hz zero-phase FIR band-pass (in place)."""
    raw.set_eeg_reference(ref_channels="average", verbose="ERROR")
    raw.filter(
        config.FMIN, config.FMAX,
        fir_design="firwin", skip_by_annotation="edge", verbose="ERROR",
    )
    return raw


def epochs_from_raw(raw: mne.io.BaseRaw, annotations: list[str], label: int):
    """Extract epochs for the given annotation descriptions; return (X, y=label)."""
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    wanted = {name: code for name, code in event_id.items() if name in annotations}
    if not wanted:
        return np.empty((0, config.N_CHANNELS, 0)), np.empty((0,), dtype=int)
    epochs = mne.Epochs(
        raw, events, event_id=wanted,
        tmin=config.TMIN, tmax=config.TMAX,
        baseline=None, picks="eeg", preload=True, verbose="ERROR",
    )
    X = epochs.get_data(copy=True).astype(np.float64)
    y = np.full(X.shape[0], label, dtype=int)
    return X, y


def build_dataset(subject: int, experiment: int):
    """Assemble (X, y) for one (subject, experiment).

    Loads + filters each distinct run-group once, then epochs per class-spec
    using its run-group's filtered raw, so exp4/5 provenance labels are correct.
    """
    specs = config.EXPERIMENTS[experiment]

    # Load+filter each distinct run-set once (keyed by the tuple of runs).
    filtered: dict[tuple, mne.io.BaseRaw] = {}
    for spec in specs:
        key = tuple(spec["runs"])
        if key not in filtered:
            filtered[key] = filter_raw(data.load_raw(subject, list(key)))

    xs, ys = [], []
    for spec in specs:
        raw = filtered[tuple(spec["runs"])]
        Xc, yc = epochs_from_raw(raw, spec["annotations"], spec["label"])
        if Xc.shape[0]:
            xs.append(Xc)
            ys.append(yc)

    if not xs:
        raise RuntimeError(f"No epochs for subject {subject}, experiment {experiment}")

    # All epochs share n_times (same TMIN/TMAX/SFREQ); concatenate.
    n_times = min(x.shape[2] for x in xs)
    xs = [x[:, :, :n_times] for x in xs]
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_preprocessing.py -v -k build_dataset`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tpv/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): filter + run-aware build_dataset for 6 experiments"
```

---

## Task 4: `tpv/csp.py` — from-scratch CSP transformer

**Files:**
- Create: `tpv/csp.py`
- Test: `tests/test_csp.py` (synthetic, fast, deterministic)

- [ ] **Step 1: Write the failing tests**

`tests/test_csp.py`:
```python
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from tpv.csp import MyCSP


def _synthetic(n_per_class=40, n_ch=8, n_times=160, seed=0):
    """Two classes: class 0 has extra variance in channel 0, class 1 in channel 1."""
    rng = np.random.default_rng(seed)
    base0 = rng.standard_normal((n_per_class, n_ch, n_times))
    base1 = rng.standard_normal((n_per_class, n_ch, n_times))
    base0[:, 0, :] *= 4.0
    base1[:, 1, :] *= 4.0
    X = np.concatenate([base0, base1], axis=0)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def test_fit_sets_learned_attributes():
    X, y = _synthetic()
    csp = MyCSP(n_components=4).fit(X, y)
    assert csp.filters_.shape == (4, X.shape[1])
    assert set(csp.classes_.tolist()) == {0, 1}


def test_transform_output_shape():
    X, y = _synthetic()
    feats = MyCSP(n_components=4).fit(X, y).transform(X)
    assert feats.shape == (X.shape[0], 4)
    assert np.isfinite(feats).all()


def test_eigenvalues_in_unit_interval():
    X, y = _synthetic()
    csp = MyCSP(n_components=4).fit(X, y)
    assert csp.eigenvalues_.min() >= -1e-9
    assert csp.eigenvalues_.max() <= 1 + 1e-9


def test_csp_separates_synthetic_classes():
    X, y = _synthetic(seed=1)
    pipe = Pipeline([("csp", MyCSP(n_components=4)),
                     ("clf", LinearDiscriminantAnalysis())])
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.95


def test_clone_compatibility():
    csp = MyCSP(n_components=6, reg=0.05)
    c2 = clone(csp)
    assert c2.get_params() == {"n_components": 6, "reg": 0.05}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_csp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tpv.csp'`.

- [ ] **Step 3: Write `tpv/csp.py`**

```python
"""From-scratch Common Spatial Patterns as a scikit-learn transformer.

Covariance estimation (numpy) + generalized eigendecomposition (scipy.linalg.eigh).
No SVD, no library CSP. See spec §4.
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class MyCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 4, reg: float = 0.01):
        # Store hyperparameters only (sklearn clone/set_params contract).
        self.n_components = n_components
        self.reg = reg

    def _class_cov(self, epochs: np.ndarray) -> np.ndarray:
        """Mean trace-normalized, symmetrized, shrinkage-regularized covariance."""
        n_ch = epochs.shape[1]
        acc = np.zeros((n_ch, n_ch))
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
        # y is required (CSP is supervised); y=None default satisfies the sklearn
        # transformer signature convention. Pipeline always passes y positionally.
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("MyCSP requires exactly 2 classes")

        c1 = self._class_cov(X[y == self.classes_[0]])
        c2 = self._class_cov(X[y == self.classes_[1]])

        # Generalized eigenproblem: C1 w = lambda (C1+C2) w. eigh -> ascending eigenvalues.
        eigvals, eigvecs = eigh(c1, c1 + c2)
        order = np.argsort(np.abs(eigvals - 0.5))[::-1]  # most discriminative first
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self.eigenvalues_ = eigvals
        self.filters_ = eigvecs.T[: self.n_components]              # (n_components, n_channels)
        self.patterns_ = np.linalg.pinv(eigvecs.T)[:, : self.n_components]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((X.shape[0], self.n_components))
        for i, E in enumerate(X):
            Z = self.filters_ @ E                # (n_components, n_times)
            var = np.var(Z, axis=1)
            total = var.sum()
            var = var / total if total > 0 else np.full_like(var, 1.0 / var.size)
            out[i] = np.log(np.clip(var, 1e-12, None))   # guard log(0) on degenerate input
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_csp.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tpv/csp.py tests/test_csp.py
git commit -m "feat(csp): from-scratch CSP transformer (numpy cov + scipy.linalg.eigh)"
```

---

## Task 5: `tpv/pipeline.py` — pipeline factory

**Files:**
- Create: `tpv/pipeline.py`
- Test: `tests/test_csp.py` (add factory tests — fast, no network)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_csp.py`:
```python
from tpv.pipeline import build_pipeline
from mne.decoding import CSP as MneCSP


def test_build_pipeline_default_is_scratch_lda():
    pipe = build_pipeline()
    assert list(dict(pipe.steps).keys()) == ["csp", "clf"]
    assert isinstance(pipe.named_steps["csp"], MyCSP)
    assert isinstance(pipe.named_steps["clf"], LinearDiscriminantAnalysis)


def test_build_pipeline_mne_csp_option():
    pipe = build_pipeline(csp="mne")
    assert isinstance(pipe.named_steps["csp"], MneCSP)


def test_build_pipeline_classifier_choices():
    for clf in ("lda", "svm", "logreg", "rf"):
        pipe = build_pipeline(clf=clf)
        assert pipe.named_steps["clf"] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_csp.py -v -k pipeline`
Expected: FAIL with `ModuleNotFoundError: No module named 'tpv.pipeline'`.

- [ ] **Step 3: Write `tpv/pipeline.py`**

```python
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


def _make_classifier(clf: str, seed):
    if clf == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if clf == "svm":
        return SVC(kernel="rbf", C=1.0, random_state=seed)
    if clf == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if clf == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed)
    raise ValueError(f"unknown classifier: {clf}")


def _make_csp(csp: str, n_components: int, reg: float):
    if csp == "scratch":
        return MyCSP(n_components=n_components, reg=reg)
    if csp == "mne":
        # Close to MyCSP's route for the parity test; the two still differ by MyCSP's
        # 0.01 identity shrinkage and component ordering, absorbed by the <0.05
        # tolerance. If parity ever fails, first match regularization (set reg=0.01),
        # do NOT relax the tolerance (spec §4.6).
        return MneCSP(n_components=n_components, reg=None, log=True,
                      norm_trace=True, cov_est="epoch")
    raise ValueError(f"unknown csp: {csp}")


def build_pipeline(csp: str = "scratch", clf: str = "lda",
                   n_components: int = 4, reg: float = 0.01, seed=None) -> Pipeline:
    return Pipeline([
        ("csp", _make_csp(csp, n_components, reg)),
        ("clf", _make_classifier(clf, seed)),
    ])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_csp.py -v -k pipeline`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tpv/pipeline.py tests/test_csp.py
git commit -m "feat(pipeline): CSP+classifier factory (scratch/mne, lda/svm/logreg/rf)"
```

---

## Task 6: CSP parity vs `mne.decoding.CSP`

**Files:**
- Create: `tests/test_csp_parity.py` (network)

- [ ] **Step 1: Write the failing test**

`tests/test_csp_parity.py`:
```python
import numpy as np
import pytest
from sklearn.model_selection import ShuffleSplit, cross_val_score
from tpv import preprocessing
from tpv.pipeline import build_pipeline


@pytest.mark.network
def test_scratch_csp_matches_mne_within_tolerance():
    X, y = preprocessing.build_dataset(1, 3)  # subject 1, imagery hands vs feet
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    acc_scratch = cross_val_score(build_pipeline(csp="scratch", clf="lda"), X, y, cv=cv).mean()
    acc_mne = cross_val_score(build_pipeline(csp="mne", clf="lda"), X, y, cv=cv).mean()

    assert abs(acc_scratch - acc_mne) < 0.05, (acc_scratch, acc_mne)
    assert acc_scratch >= 0.60   # architecture sanity on a single subject
```

- [ ] **Step 2: Run the parity guard**

Run: `.venv/bin/pytest tests/test_csp_parity.py -v`
This is a parity/integration **guard**, not a red-phase test — all its dependencies exist by Task 6, so it is **expected to PASS on first run**. If it fails on tolerance, debug MyCSP (covariance route, reg) until `|Δacc| < 0.05`; do NOT relax the tolerance to pass — fix the implementation. (To prove the test *can* fail, temporarily drop the `|λ-0.5|` ordering in MyCSP once and watch the assert trip, then revert.) Expected: PASS.

- [ ] **Step 3: If parity fails, debug (no code change if it passes)**

Checklist if `assert` fails:
- Confirm `MneCSP(cov_est="epoch", norm_trace=True, log=True)` matches MyCSP's per-trial trace-normalized route.
- Confirm `reg=0.01` keeps `C1+C2` positive-definite (no `LinAlgError`).
- Print both means; a gap > 0.05 usually means a covariance normalization mismatch, not a bug in eigh.

- [ ] **Step 4: Commit**

```bash
git add tests/test_csp_parity.py
git commit -m "test(csp): parity of from-scratch CSP vs mne.decoding.CSP (<0.05)"
```

---

## Task 7: `tpv/train.py` — train mode

**Files:**
- Create: `tpv/train.py`
- Test: `tests/test_pipeline_smoke.py` (network)

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline_smoke.py`:
```python
import os
import numpy as np
import pytest
import joblib
from tpv import train, config


@pytest.mark.network
def test_train_creates_artifact_and_returns_mean(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setenv("TPV_SEED", "42")
    mean = train.train(1, 14)            # run 14 -> exp3
    out = capsys.readouterr().out
    assert "cross_val_score:" in out
    assert 0.0 <= mean <= 1.0

    path = train.artifact_path(1, 14)
    assert path.exists()
    art = joblib.load(path)
    assert set(art.keys()) == {"pipeline", "X_test", "y_test", "meta"}
    assert art["meta"]["experiment"] == 3
    assert art["X_test"].shape[0] == art["y_test"].shape[0] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pipeline_smoke.py::test_train_creates_artifact_and_returns_mean -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tpv.train'`.

- [ ] **Step 3: Write `tpv/train.py`**

```python
"""Train mode: cross_val_score report + fit + persist artifact."""
import joblib
import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split

from tpv import config, preprocessing
from tpv.pipeline import build_pipeline


def artifact_path(subject: int, run: int):
    exp = config.RUN_TO_EXPERIMENT[run]
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
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
    joblib.dump(artifact, artifact_path(subject, run))
    return float(scores.mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_pipeline_smoke.py::test_train_creates_artifact_and_returns_mean -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tpv/train.py tests/test_pipeline_smoke.py
git commit -m "feat(train): cross_val_score report + fit + joblib artifact"
```

---

## Task 8: `tpv/predict.py` — streaming playback predict (< 2 s)

**Files:**
- Create: `tpv/predict.py`
- Test: `tests/test_pipeline_smoke.py` (add)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_smoke.py`:
```python
@pytest.mark.network
def test_predict_streams_and_reports_accuracy(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setenv("TPV_SEED", "42")
    train.train(1, 14)
    capsys.readouterr()  # clear train output

    from tpv import predict
    acc = predict.predict(1, 14)
    out = capsys.readouterr().out
    assert "epoch 00:" in out
    assert "Accuracy:" in out
    assert 0.0 <= acc <= 1.0


@pytest.mark.network
def test_predict_per_chunk_latency_under_2s(tmp_path, monkeypatch):
    import time, joblib
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setenv("TPV_SEED", "42")
    train.train(1, 14)
    art = joblib.load(train.artifact_path(1, 14))
    pipe, X_test = art["pipeline"], art["X_test"]
    t0 = time.perf_counter()
    pipe.predict(X_test[0:1])
    assert (time.perf_counter() - t0) < 2.0


def test_predict_missing_artifact_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    from tpv import predict
    with pytest.raises(FileNotFoundError):
        predict.predict(1, 14)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline_smoke.py -v -k predict`
Expected: FAIL with `ModuleNotFoundError: No module named 'tpv.predict'`.

- [ ] **Step 3: Write `tpv/predict.py`**

```python
"""Predict mode: replay persisted held-out epochs as a simulated stream (<2s/chunk)."""
import time
import joblib
import numpy as np

from tpv import config
from tpv.train import artifact_path

LATENCY_BUDGET_S = 2.0


def predict(subject: int, run: int) -> float:
    path = artifact_path(subject, run)
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `train {subject} {run}` first.")

    art = joblib.load(path)                       # heavy work BEFORE the loop
    pipe, X_test, y_test = art["pipeline"], art["X_test"], art["y_test"]

    print("epoch nb: [prediction] [truth] equal?")
    correct = 0
    for i in range(X_test.shape[0]):
        chunk = X_test[i:i + 1]                   # (1, 64, n_times)
        t0 = time.perf_counter()
        pred = int(pipe.predict(chunk)[0])        # CSP transform + LDA dot product
        latency = time.perf_counter() - t0
        assert latency < LATENCY_BUDGET_S, f"chunk {i} took {latency:.3f}s"

        truth = int(y_test[i])
        equal = pred == truth
        correct += equal
        # Map internal class index {0,1} -> {1,2} for parity with the PDF example.
        # (predict is single-run addressable -> only exp0-3, where 0/1 == T1/T2.)
        print(f"epoch {i:02d}:    [{pred + 1}]    [{truth + 1}] {equal}")

    acc = correct / X_test.shape[0] if X_test.shape[0] else 0.0
    print(f"Accuracy: {acc:.4f}")
    return float(acc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline_smoke.py -v -k predict`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tpv/predict.py tests/test_pipeline_smoke.py
git commit -m "feat(predict): streaming playback prediction with <2s latency guard"
```

---

## Task 9: `tpv/evaluate.py` — classifier comparison & run_all

**Files:**
- Create: `tpv/evaluate.py`
- Test: `tests/test_pipeline_smoke.py` (add a 1-subject run_all test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_smoke.py`:
```python
@pytest.mark.network
def test_run_all_single_subject_returns_grand_mean(monkeypatch, capsys):
    monkeypatch.setenv("TPV_SEED", "42")
    from tpv import evaluate
    grand = evaluate.run_all(subjects=[1], experiments=range(6))
    out = capsys.readouterr().out
    assert "experiment 0:" in out
    assert "Mean accuracy of 6 experiments" in out
    assert 0.0 <= grand <= 1.0


@pytest.mark.network
def test_compare_classifiers_returns_all(monkeypatch):
    from tpv import evaluate
    scores = evaluate.compare_classifiers(1, 3)
    assert set(scores.keys()) == {"lda", "svm", "logreg", "rf"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline_smoke.py -v -k "run_all or compare"`
Expected: FAIL with `ModuleNotFoundError: No module named 'tpv.evaluate'`.

- [ ] **Step 3: Write `tpv/evaluate.py`**

```python
"""Cross-validated accuracy, classifier comparison, and the full 109x6 run."""
import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_val_score

from tpv import config, preprocessing
from tpv.pipeline import build_pipeline

CLASSIFIERS = ("lda", "svm", "logreg", "rf")


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline_smoke.py -v -k "run_all or compare"`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tpv/evaluate.py tests/test_pipeline_smoke.py
git commit -m "feat(evaluate): cross-val accuracy, classifier comparison, run_all"
```

---

## Task 10: `mybci.py` — CLI dispatcher

**Files:**
- Create: `mybci.py`
- Test: `tests/test_cli.py` (fast — mock the behaviors)

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
import pytest
import mybci


def test_no_args_calls_run_all(monkeypatch):
    called = {}
    monkeypatch.setattr(mybci, "run_all", lambda: called.setdefault("ran", True))
    assert mybci.main([]) == 0
    assert called.get("ran")


def test_train_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(mybci, "train", lambda s, r: seen.update(s=s, r=r))
    assert mybci.main(["4", "14", "train"]) == 0
    assert seen == {"s": 4, "r": 14}


def test_predict_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(mybci, "predict", lambda s, r: seen.update(s=s, r=r))
    assert mybci.main(["4", "14", "predict"]) == 0
    assert seen == {"s": 4, "r": 14}


def test_invalid_subject_errors():
    assert mybci.main(["200", "14", "train"]) != 0


def test_invalid_run_errors():
    assert mybci.main(["4", "2", "train"]) != 0     # run 2 is baseline, not a task


def test_bad_arity_errors():
    assert mybci.main(["4", "14"]) != 0             # missing mode
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mybci'`.

- [ ] **Step 3: Write `mybci.py`**

```python
#!/usr/bin/env python
"""Total Perspective Vortex — EEG BCI CLI.

Usage:
  python mybci.py                      run all 6 experiments x 109 subjects
  python mybci.py <subject> <run> train
  python mybci.py <subject> <run> predict
"""
import sys

from tpv.train import train
from tpv.predict import predict
from tpv.evaluate import run_all

USAGE = (
    "usage:\n"
    "  python mybci.py                      # all 6 experiments x 109 subjects\n"
    "  python mybci.py <subject 1-109> <run 3-14> train\n"
    "  python mybci.py <subject 1-109> <run 3-14> predict\n"
)


def _fail(msg: str) -> int:
    print(msg, file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if len(argv) == 0:
        run_all()
        return 0

    if len(argv) != 3:
        return _fail("error: expected `<subject> <run> <train|predict>`")

    s_str, r_str, mode = argv
    try:
        subject, run = int(s_str), int(r_str)
    except ValueError:
        return _fail("error: subject and run must be integers")

    if not (1 <= subject <= 109):
        return _fail(f"error: subject {subject} out of range 1..109")
    if not (3 <= run <= 14):
        return _fail(f"error: run {run} out of range 3..14 (1/2 are baseline)")
    if mode not in ("train", "predict"):
        return _fail(f"error: mode must be train|predict, got {mode!r}")

    if mode == "train":
        train(subject, run)
    else:
        predict(subject, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Manual end-to-end smoke (network)**

Run:
```bash
.venv/bin/python mybci.py 1 14 train
.venv/bin/python mybci.py 1 14 predict
```
Expected: train prints a 10-value array + `cross_val_score: <mean>`; predict prints `epoch 00:`…`Accuracy:`.

- [ ] **Step 6: Commit**

```bash
git add mybci.py tests/test_cli.py
git commit -m "feat(cli): argparse dispatcher for train/predict/run_all"
```

---

## Task 11: `tpv/viz.py` — visualizations (lazy matplotlib)

**Files:**
- Create: `tpv/viz.py`
- Test: covered by the smoke run below (plotting is hard to unit-test; verify it imports and produces figures without crashing).

- [ ] **Step 1: Write `tpv/viz.py`**

```python
"""Optional raw/PSD/CSP-pattern visualizations. matplotlib imported lazily."""
from tpv import config, data, preprocessing


def plot_raw_before_after(subject: int = 1, runs=(6, 10, 14), show: bool = True):
    """Plot raw EEG before and after the 7-30 Hz band-pass (defense visual)."""
    import matplotlib.pyplot as plt  # noqa: F401  (lazy)

    raw = data.load_raw(subject, list(runs))
    fig_before = raw.copy().plot(show=show, title="Raw (unfiltered)")
    fig_after = preprocessing.filter_raw(raw.copy()).plot(
        show=show, title=f"Filtered {config.FMIN}-{config.FMAX} Hz")
    return fig_before, fig_after


def plot_psd_before_after(subject: int = 1, runs=(6, 10, 14), show: bool = True):
    raw = data.load_raw(subject, list(runs))
    fig_before = raw.copy().compute_psd(method="welch").plot(show=show)
    fig_after = preprocessing.filter_raw(raw.copy()).compute_psd(method="welch").plot(show=show)
    return fig_before, fig_after


def plot_csp_patterns(subject: int = 1, experiment: int = 3, show: bool = True):
    """Show the first CSP spatial pattern over the scalp (C3/Cz/C4 sensorimotor focus)."""
    import mne
    from tpv.csp import MyCSP

    import matplotlib.pyplot as plt

    raw = preprocessing.filter_raw(data.load_raw(subject, config.runs_for_experiment(experiment)))
    raw.pick("eeg")  # info now carries the 64-channel montage set in load_raw
    X, y = preprocessing.build_dataset(subject, experiment)
    csp = MyCSP(n_components=config.N_COMPONENTS).fit(X, y)
    # mne.viz.plot_topomap returns (im, cn) = (AxesImage, ContourSet), NOT a Figure.
    # Pass an explicit axes and return its owning Figure (matches sibling plot fns).
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(csp.patterns_[:, 0], raw.info, axes=ax, show=show)
    return fig
```

- [ ] **Step 2: Smoke-run (network, optional; skip if headless)**

Run:
```bash
.venv/bin/python -c "from tpv import viz; viz.plot_raw_before_after(show=False); print('viz ok')"
```
Expected: `viz ok` (figures built without showing). If a montage/topomap call errors, simplify `plot_csp_patterns` — the raw/PSD plots are the required defense visuals; CSP topomap is a nice-to-have.

- [ ] **Step 3: Commit**

```bash
git add tpv/viz.py
git commit -m "feat(viz): raw/PSD before-after and CSP pattern plots (lazy matplotlib)"
```

---

## Task 12: `scripts/validate_60.py` — the ≥ 60 % gate

**Files:**
- Create: `scripts/validate_60.py`

- [ ] **Step 1: Write `scripts/validate_60.py`**

```python
#!/usr/bin/env python
"""Pre-defense gate: run_all and exit non-zero if the grand mean < 0.60.

  .venv/bin/python scripts/validate_60.py            # full 109x6 (slow)
  .venv/bin/python scripts/validate_60.py --fast 5   # first N subjects
"""
import argparse
import sys

from tpv import config
from tpv.evaluate import run_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", type=int, default=0,
                    help="evaluate only the first N subjects (0 = all 109)")
    args = ap.parse_args()

    subjects = range(1, (args.fast or config.N_SUBJECTS) + 1)
    grand = run_all(subjects=subjects, experiments=range(6), fast=bool(args.fast))

    if grand < 0.60:
        print(f"FAIL: grand mean {grand:.4f} < 0.60", file=sys.stderr)
        return 1
    print(f"PASS: grand mean {grand:.4f} >= 0.60")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Fast smoke (network)**

Run:
```bash
.venv/bin/python scripts/validate_60.py --fast 3
```
Expected: prints per-experiment means + grand mean; exits 0 if ≥ 0.60. (3 subjects is a smoke check, not the official gate.)

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_60.py
git commit -m "feat(scripts): validate_60 pre-defense accuracy gate"
```

---

## Task 13: `README.md` & final full-gate run

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# Total Perspective Vortex

EEG brain-computer interface that classifies PhysioNet motor-imagery/execution
trials with a from-scratch CSP transformer inside a scikit-learn pipeline.

## Setup
```bash
/Users/keokim/.brew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
The PhysioNet `eegmmidb` dataset auto-downloads to `~/mne_data` on first run.

## Usage
```bash
python mybci.py 4 14 train     # subject 4, run 14 -> experiment 3 (imagery hands vs feet)
python mybci.py 4 14 predict   # replay held-out epochs as a stream
python mybci.py                # all 6 experiments x 109 subjects, grand mean
```

`<run>` maps to an experiment: 3/7/11→exp0, 4/8/12→exp1, 5/9/13→exp2, 6/10/14→exp3.
Experiments 4/5 (real vs imagined) are evaluated only by the no-argument full run.

## The 6 experiments
| Exp | Runs | Task |
|-----|------|------|
| 0 | 3,7,11 | execution L vs R hand |
| 1 | 4,8,12 | imagery L vs R hand |
| 2 | 5,9,13 | execution hands vs feet |
| 3 | 6,10,14 | imagery hands vs feet |
| 4 | 3,7,11 vs 4,8,12 | execution vs imagery (hands) |
| 5 | 5,9,13 vs 6,10,14 | execution vs imagery (bilateral) |

## Reproducibility
`TPV_SEED=42 python mybci.py 4 14 train` for a deterministic run; unset = different
splits each time (per subject requirement). The ≥ 60 % bar applies to the full-run
grand mean, not to a single `train` call.

## Tests
```bash
.venv/bin/pytest -m "not network"     # fast unit tests
.venv/bin/pytest                      # full suite (downloads subject 1)
.venv/bin/python scripts/validate_60.py   # ≥ 60 % gate (slow)
```
````

- [ ] **Step 2: Run the fast test suite**

Run: `.venv/bin/pytest -m "not network" -v`
Expected: all fast tests PASS (config, csp, pipeline, cli).

- [ ] **Step 3: Run the full network suite once**

Run: `.venv/bin/pytest -v`
Expected: all tests PASS (parity < 0.05, train/predict/run_all smoke).

- [ ] **Step 4: Run the official ≥ 60 % gate (full 109 subjects — slow, tens of minutes)**

Run: `.venv/bin/python scripts/validate_60.py`
Expected: `PASS: grand mean <X> >= 0.60`. If it fails, the spec's margin-management levers apply (re-check label correctness per run-group, confirm no leakage, consider classifier per the comparison) — investigate, do not lower the gate.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: README with usage, experiments, reproducibility"
```

---

## Self-review notes (spec coverage)

- Custom transformer (`BaseEstimator`+`TransformerMixin`): Task 4. ✓
- sklearn `Pipeline` + `cross_val_score` on the whole pipeline: Tasks 5, 7, 9. ✓
- Anti-leakage (CSP inside pipeline, refit per fold): Tasks 5, 7, 9 (CSP is step 0). ✓
- From-scratch CSP via numpy cov + `scipy.linalg.eigh`, validated vs MNE: Tasks 4, 6. ✓
- 6 experiments with run-aware T1/T2 labels + exp4/5 provenance: Tasks 1, 3. ✓
- Streaming predict < 2 s, no mne-realtime: Task 8. ✓
- Three CLI forms: Task 10. ✓
- ≥ 60 % grand-mean gate (run_all only): Tasks 9, 12, 13. ✓
- Reproducibility / seed / requirements / .gitignore: Tasks 0, 1. ✓
- Visualizations for defense: Task 11. ✓
- Deferred (full bonus): not implemented by design (spec §13).
```
