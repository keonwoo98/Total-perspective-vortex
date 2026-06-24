"""Load PhysioNet eegmmidb EDF runs and assemble one Raw per run-group."""
import mne
from mne.datasets import eegbci

from tpv import config


def load_raw(subject: int, runs: list[int]) -> mne.io.BaseRaw:
    """Download (if needed), read, standardize, concatenate, montage, and
    sample-rate-guard the given runs for one subject.

    Returns a single concatenated Raw with EEG channels and T0/T1/T2 annotations.
    """
    # Download into the project-local mne_data/ (config.DATA_DIR) instead of ~/mne_data,
    # so the data travels with the project folder. update_path=False keeps MNE's global
    # config untouched. Pass subject/runs POSITIONALLY: the first param was renamed
    # subject->subjects in MNE 1.9, so positional works across the pinned 1.6-1.12 range.
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = eegbci.load_data(subject, runs, path=str(config.DATA_DIR), update_path=False)
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
