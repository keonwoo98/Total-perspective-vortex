#!/usr/bin/env python
"""Pre-defense gate: run_all and exit non-zero if the grand mean < 0.60.

  .venv/bin/python scripts/validate_60.py            # full 109x6 (slow)
  .venv/bin/python scripts/validate_60.py --fast 5   # first N subjects
"""
import argparse
import sys

from tpv import config
from tpv.evaluate import run_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", type=int, default=0,
                    help="evaluate only the first N subjects (0 = all 109)")
    args = ap.parse_args()

    subjects = range(1, (args.fast or config.N_SUBJECTS) + 1)
    grand = run_all(subjects=subjects, experiments=range(6), fast=bool(args.fast))

    if grand < 0.60:
        print(f"FAIL: grand mean {grand:.4f} < 0.60", file=sys.stderr)
        return 1
    print(f"PASS: grand mean {grand:.4f} >= 0.60")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
