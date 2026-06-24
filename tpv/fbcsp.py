"""Minimal Filter-Bank CSP (FBCSP): per-sub-band CSP features, concatenated.

Band-passes the epochs into a few fixed sub-bands (inside the existing 7-30 Hz passband),
fits a MyCSP per band, and concatenates the per-band log-variance features. A more complex
dimensionality-reduction algorithm than plain CSP. Band-pass lives inside fit/transform so
cross_val_score refits per fold (no leakage).
"""
import numpy as np
from mne.filter import filter_data
from sklearn.base import BaseEstimator, TransformerMixin

from tpv import config
from tpv.csp import MyCSP

DEFAULT_BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 30.0))


class FilterBankCSP(TransformerMixin, BaseEstimator):
    def __init__(self, bands=DEFAULT_BANDS, sfreq=config.SFREQ, n_components=2, reg=config.REG):
        # Store hyperparameters only (sklearn clone/set_params contract).
        self.bands = bands
        self.sfreq = sfreq
        self.n_components = n_components
        self.reg = reg

    def _bandpass(self, X, lo, hi):
        # Filters along the last axis (time); X is (n_epochs, n_channels, n_times).
        return filter_data(X, self.sfreq, lo, hi, method="fir", verbose="ERROR")

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_times = X.shape[2]
        # Fail loudly if epochs are too short for the lowest band's FIR filter.
        lo_min = min(lo for lo, hi in self.bands)
        trans = max(lo_min * 0.25, 2.0)                  # mne 'auto' low transition bandwidth
        fir_len = int(round(3.3 * self.sfreq / trans))   # mne firwin 'auto' length heuristic
        if fir_len >= n_times:
            raise ValueError(
                f"FilterBankCSP: FIR length ~{fir_len} >= n_times {n_times}; "
                f"epochs too short for the {lo_min} Hz band")
        self.csps_ = []
        for lo, hi in self.bands:
            csp = MyCSP(n_components=self.n_components, reg=self.reg).fit(self._bandpass(X, lo, hi), y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        feats = [csp.transform(self._bandpass(X, lo, hi))
                 for (lo, hi), csp in zip(self.bands, self.csps_)]
        return np.concatenate(feats, axis=1)
