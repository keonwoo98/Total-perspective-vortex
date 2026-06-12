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
    import matplotlib.pyplot as plt
    import mne

    from tpv.csp import MyCSP

    raw = preprocessing.filter_raw(data.load_raw(subject, config.runs_for_experiment(experiment)))
    raw.pick("eeg")  # info now carries the 64-channel montage set in load_raw
    X, y = preprocessing.build_dataset(subject, experiment)
    csp = MyCSP(n_components=config.N_COMPONENTS).fit(X, y)
    # mne.viz.plot_topomap returns (im, cn) = (AxesImage, ContourSet), NOT a Figure.
    # Pass an explicit axes and return its owning Figure (matches sibling plot fns).
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(csp.patterns_[:, 0], raw.info, axes=ax, show=show)
    return fig
