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
