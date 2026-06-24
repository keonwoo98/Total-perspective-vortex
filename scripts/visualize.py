#!/usr/bin/env python
"""Visualize raw vs filtered EEG — the preprocessing defense plots.

  python scripts/visualize.py          # subject 1, experiment 3 runs (6/10/14)
  python scripts/visualize.py 4 14     # subject 4, the run-group for run 14

Shows the raw signal, the 7-30 Hz filtered signal, and their PSDs (so the
filtered band is visibly cleaner). Close the plot windows to exit.
"""
import argparse
import sys
from pathlib import Path

# Make `tpv` importable when run as `python scripts/visualize.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tpv import config, viz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", nargs="?", type=int, default=1, help="subject 1-109")
    ap.add_argument("run", nargs="?", type=int, default=6, help="run 3-14 (selects its experiment)")
    args = ap.parse_args()

    runs = config.runs_for_experiment(config.RUN_TO_EXPERIMENT[args.run])
    print(f"Visualizing subject {args.subject}, runs {runs} — raw vs {config.FMIN}-{config.FMAX} Hz filtered...")

    viz.plot_raw_before_after(subject=args.subject, runs=runs, show=False)
    viz.plot_psd_before_after(subject=args.subject, runs=runs, show=False)

    import matplotlib.pyplot as plt
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
