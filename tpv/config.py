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

# Project-local EEG cache (anchored to the repo root, so it travels with the
# project folder). gitignored. Override default ~/mne_data location.
DATA_DIR = Path(__file__).resolve().parent.parent / "mne_data"

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
