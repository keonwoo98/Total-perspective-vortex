import numpy as np
import pytest
from tpv import config, data
from tpv import preprocessing


@pytest.mark.network
def test_load_raw_subject1_exp3_runs():
    raw = data.load_raw(1, [6, 10, 14])
    assert raw.info["sfreq"] == config.SFREQ
    assert len(raw.get_data(picks="eeg")) == config.N_CHANNELS
    # annotations T0/T1/T2 survived concatenation
    descs = set(raw.annotations.description)
    assert {"T0", "T1", "T2"}.issubset(descs)


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
