"""Optional second dataset (bonus): BCI Competition IV-2a via moabb.

load_external(subject) returns (X, y) in the same (n_epochs, n_channels, n_times) float64 /
int{0,1} contract as preprocessing.build_dataset, so the existing CSP+classifier pipeline runs
unchanged (MyCSP is channel-count-agnostic). moabb is an OPTIONAL dependency (requirements-bonus.txt),
lazily imported here so the mandatory project works without it. This is an INDEPENDENT run on a
different dataset (22 channels, 250 Hz) — not merged with the PhysioNet data.
"""
import numpy as np

from tpv import config

_LABELS = {"left_hand": 0, "right_hand": 1}


def load_external(subject: int = 1):
    """Load BNCI2014_001 (BCI IV-2a) left-vs-right-hand motor imagery for one subject.

    Returns (X, y): X float64 (n_epochs, n_channels, n_times), y int in {0, 1}.
    Requires the optional moabb dependency: pip install -r requirements-bonus.txt
    """
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import LeftRightImagery
    except ImportError as exc:
        raise ImportError(
            "load_external needs the optional 'moabb' dependency: "
            "pip install -r requirements-bonus.txt"
        ) from exc

    paradigm = LeftRightImagery(fmin=config.FMIN, fmax=config.FMAX)
    X, y_str, _ = paradigm.get_data(dataset=BNCI2014_001(), subjects=[subject], return_epochs=False)
    X = np.asarray(X, dtype=np.float64)
    y = np.array([_LABELS[label] for label in y_str], dtype=int)
    return X, y
