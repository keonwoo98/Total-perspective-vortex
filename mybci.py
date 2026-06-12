#!/usr/bin/env python
"""Total Perspective Vortex - EEG BCI CLI.

Usage:
  python mybci.py                      run all 6 experiments x 109 subjects
  python mybci.py <subject> <run> train
  python mybci.py <subject> <run> predict
"""
import sys

from tpv.train import train
from tpv.predict import predict
from tpv.evaluate import run_all

USAGE = (
    "usage:\n"
    "  python mybci.py                      # all 6 experiments x 109 subjects\n"
    "  python mybci.py <subject 1-109> <run 3-14> train\n"
    "  python mybci.py <subject 1-109> <run 3-14> predict\n"
)


def _fail(msg: str) -> int:
    print(msg, file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if len(argv) == 0:
        run_all()
        return 0

    if len(argv) != 3:
        return _fail("error: expected `<subject> <run> <train|predict>`")

    s_str, r_str, mode = argv
    try:
        subject, run = int(s_str), int(r_str)
    except ValueError:
        return _fail("error: subject and run must be integers")

    if not (1 <= subject <= 109):
        return _fail(f"error: subject {subject} out of range 1..109")
    if not (3 <= run <= 14):
        return _fail(f"error: run {run} out of range 3..14 (1/2 are baseline)")
    if mode not in ("train", "predict"):
        return _fail(f"error: mode must be train|predict, got {mode!r}")

    if mode == "train":
        train(subject, run)
    else:
        predict(subject, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
