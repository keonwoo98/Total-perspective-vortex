"""Band-pass filtering and run-aware epoching into (X, y) datasets."""
import warnings

import mne
import numpy as np

from tpv import config, data


def filter_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Average reference + 7-30 Hz zero-phase FIR band-pass.

    Mutates the passed raw in place and returns it. Do NOT call twice on the
    same raw, or the band-pass would be applied twice.
    """
    raw.set_eeg_reference(ref_channels="average", verbose="ERROR")
    raw.filter(
        config.FMIN, config.FMAX,
        fir_design="firwin", skip_by_annotation="edge", verbose="ERROR",
    )
    return raw


def epochs_from_raw(
    raw: mne.io.BaseRaw, annotations: list[str], label: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract epochs for the given annotation descriptions; return (X, y=label)."""
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    # Keep only the requested annotation types; T0 (rest) and any BAD_/edge
    # boundary descriptions added by concatenate_raws are dropped here.
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


def build_dataset(subject: int, experiment: int) -> tuple[np.ndarray, np.ndarray]:
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
    lengths = [x.shape[2] for x in xs]
    if len(set(lengths)) > 1:
        warnings.warn(f"n_times mismatch {lengths}; truncating to {min(lengths)}")
    n_times = min(lengths)
    xs = [x[:, :, :n_times] for x in xs]
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y
