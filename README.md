# Total Perspective Vortex

EEG brain-computer interface that classifies PhysioNet motor-imagery/execution
trials with a from-scratch CSP transformer inside a scikit-learn pipeline.

## Setup
```bash
/Users/keokim/.brew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
The PhysioNet `eegmmidb` dataset auto-downloads to `~/mne_data` on first run.

## Usage
Run all commands from the repository root.
```bash
.venv/bin/python mybci.py 4 14 train     # subject 4, run 14 -> experiment 3 (imagery hands vs feet)
.venv/bin/python mybci.py 4 14 predict   # replay held-out epochs as a stream
.venv/bin/python mybci.py                # all 6 experiments x 109 subjects, grand mean
```

`<run>` maps to an experiment: 3/7/11->exp0, 4/8/12->exp1, 5/9/13->exp2, 6/10/14->exp3.
Experiments 4/5 (real vs imagined) are evaluated only by the no-argument full run.

## The 6 experiments
| Exp | Runs | Task |
|-----|------|------|
| 0 | 3,7,11 | execution L vs R hand |
| 1 | 4,8,12 | imagery L vs R hand |
| 2 | 5,9,13 | execution hands vs feet |
| 3 | 6,10,14 | imagery hands vs feet |
| 4 | 3,7,11 vs 4,8,12 | execution vs imagery (hands) |
| 5 | 5,9,13 vs 6,10,14 | execution vs imagery (bilateral) |

## Reproducibility
`TPV_SEED=42 .venv/bin/python mybci.py 4 14 train` for a deterministic run; unset =
different splits each time (per subject requirement). The >= 60% bar applies to the
full-run grand mean, not to a single `train` call.

## Tests
```bash
.venv/bin/pytest -m "not network"     # fast unit tests (config, csp, pipeline, cli)
.venv/bin/pytest                      # full suite (downloads subject 1)
.venv/bin/python scripts/validate_60.py            # >= 60% gate, full 109 subjects (slow)
.venv/bin/python scripts/validate_60.py --fast 5   # quick 5-subject preview
```
Run `validate_60.py` from the repo root (so `tpv` is importable).
