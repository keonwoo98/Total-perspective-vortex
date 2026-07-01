"""Optional second dataset (bonus): BCI Competition IV-2a via moabb.

load_external(subject) returns (X, y) in the same (n_epochs, n_channels, n_times) float64 /
int{0,1} contract as preprocessing.build_dataset, so the existing CSP+classifier pipeline runs
unchanged (MyCSP is channel-count-agnostic). moabb is in requirements.txt; it is lazily imported
here so importing tpv never loads moabb's heavy tree unless load_external is called. This is an INDEPENDENT run on a
different dataset (22 channels, 250 Hz) — not merged with the PhysioNet data.
"""
import warnings

import numpy as np

from tpv import config

_LABELS = {"left_hand": 0, "right_hand": 1}

# moabb resolves the BNCI cache from the dataset-specific key first, falling back to MNE_DATA.
# We point both at DATA_DIR for the call and restore them after (no persistent side effect).
_MNE_PATH_KEYS = ("MNE_DATASETS_BNCI_PATH", "MNE_DATA")


def load_external(subject: int = 1):
    """Load BNCI2014_001 (BCI IV-2a) left-vs-right-hand motor imagery for one subject.

    Returns (X, y): X float64 (n_epochs, n_channels, n_times), y int in {0, 1}.
    Requires moabb (listed in requirements.txt).

    Downloads into the project-local DATA_DIR (mne_data/), like PhysioNet, so the bonus
    dataset travels with the project folder. moabb reads its cache location from MNE config
    keys; we point those at DATA_DIR only for the duration of this call and restore the
    user's previous values afterwards (no persistent global side effect).
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from mne import get_config, set_config
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import LeftRightImagery
    except ImportError as exc:
        raise ImportError(
            "load_external needs 'moabb' (in requirements.txt): pip install -r requirements.txt"
        ) from exc

    # moabb's dataset key isn't in MNE's known list, so set_config emits a harmless
    # "Setting non-standard config type" RuntimeWarning; silence just that message.
    def _quiet_set(k, v):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Setting non-standard config type", category=RuntimeWarning)
            set_config(k, v)

    target = str(config.DATA_DIR)
    prev = {k: get_config(k) for k in _MNE_PATH_KEYS}
    for k in _MNE_PATH_KEYS:
        _quiet_set(k, target)
    try:
        paradigm = LeftRightImagery(fmin=config.FMIN, fmax=config.FMAX)
        X, y_str, _ = paradigm.get_data(
            dataset=BNCI2014_001(), subjects=[subject], return_epochs=False)
    finally:
        for k, v in prev.items():
            _quiet_set(k, v)  # restore (None removes the key)

    X = np.asarray(X, dtype=np.float64)
    y = np.array([_LABELS[label] for label in y_str], dtype=int)
    return X, y
