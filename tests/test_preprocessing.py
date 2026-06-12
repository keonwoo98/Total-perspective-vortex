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
