# Total Perspective Vortex — Defense Q&A (English)

> A robust, defense-ready walkthrough organized around the **official evaluation sheet**.
> For every criterion you get: **📋 what it checks → 🖥️ what to demonstrate → 🗣️ what to say → ❓ a deep Q&A bank** (routine *and* tough/adversarial questions an examiner may ask).
> **Part 2** is a concept primer so you can answer any follow-up. **Part 6** is a cross-cutting hard-question bank.
> The evaluation is conducted in English — every answer below is phrased so you can say it almost verbatim.
> Measured numbers come from this exact codebase (PhysioNet eegmmidb, MNE + scikit-learn).

---

## How to use this document

1. Skim **Part 1** (pitch + map) and rehearse the 60-second pitch.
2. Read **Part 2** (concepts) until each idea is solid — that is what makes you bullet-proof in Q&A.
3. Walk **Part 3** top to bottom during the defense (it follows the evaluation sheet order).
4. **Part 6** is your "curveball" insurance — the hardest questions and how to answer them.

Legend: 🗣️ = say this, 🖥️ = run/show this, 👉 = point at the screen, ❓ = question + answer.

---

# 0. Before the defense (do once)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fetch datasets into the PROJECT folder + confirm the 60% gate. Slow the first time.
python scripts/validate_60.py            # full 109 subjects → grand mean 0.658
python scripts/bonus_demo.py             # pulls the bonus dataset (BCI IV-2a) into ./mne_data too
```

- Both datasets live **inside the project** (`./mne_data/`): PhysioNet (~3.1 GB) and BCI IV-2a (~83 MB). They travel with the folder. `./mne_data/` is gitignored, so only code is committed.
- On the defense machine, run the two commands above once beforehand (internet required for the first download).

---

# 1. Sixty-second pitch + project map

**Pitch (say this):**
> "This is an EEG brain–computer interface. From PhysioNet motor-movement/imagery recordings (109 subjects, 64 channels), I classify which action a subject performed or imagined — left vs right hand, hands vs feet, real vs imagined — using **only the brain signal**. The pipeline is three stages: **(1) preprocessing** — average reference, a 7–30 Hz band-pass, and 2-second epoching; **(2) dimensionality reduction** — a **from-scratch CSP** that turns 64 channels into 4 discriminative numbers; **(3) classification** — LDA. I score it honestly with scikit-learn's `cross_val_score`. The mandatory requirement is a **from-scratch dimensionality-reduction algorithm integrated into sklearn**, and a **mean accuracy ≥ 60%** across the six experiment types over all 109 subjects. I reach **0.658**, and I implemented all five bonuses."

**Map:**
```
1 EEG table (64ch × 160Hz) → 2 goal (6 binary tasks, ≥60%) → 3 channels (C3/C4) →
4 epoching (≈45 trials) → 5 ERD (contralateral idle-rhythm drop) → 6 filter (7–30Hz) →
7 CSP (64×321 → 4 numbers) → 8 LDA (4 numbers → decision) → 9 cross-val + leakage control + 60% gate
```

---

# 2. Concept primer (so you can answer anything)

### 2.1 EEG = 64 microphones on the scalp
64 electrodes, each sampling voltage **160 times per second**, in **microvolts (µV)**. One electrode = one wavy line; 64 together = a **(64 × time)** table. One ~125 s recording is **(64, 20000)** because `125 s × 160 Hz = 20000` samples.

### 2.2 Channels and C3/C4 (a fact to accept)
Electrode names encode scalp position: first letter = region (`C` = central/**motor strip**), digit = side (**odd = left, even = right, z = midline**). So **C3** (left motor cortex) ↔ **right hand**, **C4** (right motor cortex) ↔ **left hand** — the brain controls the **contralateral** side. The 7 `C*` channels are where hand/foot signal is strongest.
**"Why not just use C3/C4?"** Each of the 6 tasks hides its signal in different channels (hands → C3/C4, feet → Cz). Measured (10-subject mean): C3 alone ≈ 0.48, C3+C4 ≈ 0.57, but on **hands-vs-feet** all-64+CSP = **0.71** vs 0.56 for two channels. So we don't hard-pick channels — **CSP finds the best channel mixture per task automatically.**

### 2.3 The goal = six **binary** guessing games, averaged ≥ 60%
Each task is still a 2-class choice; only *which two* changes:

| Exp | runs | what vs what |
|---|---|---|
| 0 | 3,7,11 | executed **left vs right hand** |
| 1 | 4,8,12 | imagined **left vs right hand** |
| 2 | 5,9,13 | executed **hands vs feet** |
| 3 | 6,10,14 | imagined **hands vs feet** |
| 4 | 3,7,11 vs 4,8,12 | **executed vs imagined** (hands) |
| 5 | 5,9,13 vs 6,10,14 | **executed vs imagined** (hands/feet) |

Score = for each experiment, average over 109 subjects (6 means) → **mean of those 6 ≥ 0.60**. We get **0.658**.

### 2.4 Epoching, and why "321"
The data ships with **annotations**: `T0` = rest, `T1`/`T2` = the two actions (their meaning is fixed per run group). We drop rest, and **cut a 2-second window at each T1/T2** → one trial = **(64 channels × 321 samples)**.
**Why 321 = 2 × 160 + 1?** Fence-post counting: sampling from t=0 to t=2.0 inclusive gives **320 intervals but 321 points** (the endpoint at t=0 adds one). One subject·experiment ≈ **45 trials** (3 runs × ~15), labels ~balanced (e.g. 23/22).

### 2.5 ERD — why this is even possible
A resting motor region oscillates strongly at 8–30 Hz (neurons fire **in sync**, like a stadium clap). **Using** that region (moving/imagining) **desynchronizes** it → the oscillation **shrinks**. That's **ERD (Event-Related Desynchronization)**, and it's **contralateral**: right hand → C3 quiets; left hand → C4 quiets.
Measured (20-subject mean, µV²): left hand C3=83.1 / C4=**78.7**; right hand C3=**82.2** / C4=89.3 — the contralateral pattern is real but **subtle**, which is exactly why we need CSP+LDA to amplify it. **Key point: the answer lives in the *size* of the oscillation (its variance), not the raw waveform.**

### 2.6 Filtering = average reference + 7–30 Hz FIR (firwin)
1. **Average reference**: subtract the 64-channel mean at each instant → removes signal common to all electrodes (like subtracting room murmur to hear each voice).
2. **7–30 Hz band-pass**: ERD lives in **mu (8–12 Hz) + beta (13–30 Hz)**. Below 7 Hz = slow drift; above 30 Hz = muscle/50–60 Hz line noise — both discarded.
3. **firwin** = the recipe for a **FIR** filter (each output sample = a weighted average of neighbors). It is **zero-phase** (doesn't shift timing) and always stable. The cutoff is a steep slope, not a vertical cliff.
4. We filter the **long continuous signal first**, then cut epochs (filtering tiny 2 s windows corrupts the edges).
Measured band power at C3 (before → after): 0–7 Hz **3743 → 8**, 7–30 Hz **913 → 164**, 30–80 Hz **162 → 7**. So the ERD band, previously buried under 4× larger drift, becomes the dominant survivor.

### 2.7 CSP in 3 steps (the from-scratch core: `tpv/csp.py`)
**Goal:** turn one trial (64 × 321 = ~20k numbers) into **4 numbers** that separate the two classes.

**(1) Why reduce.** The answer is only in oscillation *size*, and with ~45 trials you cannot fit a boundary in 20k-D (overfitting). 4 numbers make a clean boundary possible.

**(2) How it mixes — a "recipe" and `@`.** A **spatial filter** = 64 weights (one per channel). Apply it to a trial with matrix multiply `@` = "**multiply-and-sum**": `filters @ E` blends 64 channels into one **virtual channel**. Two reductions happen: `64×321 →(@)→ 4×321 →(variance per row)→ 4 numbers` — first fewer channels, then time collapses to oscillation size. (Recipe #1's top weights land on FC4/C4/C6/FC6/CP4 — all **right motor cortex** — i.e. CSP built a "right-motor detector".)

**(3) How the filters are chosen — covariance + eigendecomposition.** Build each class's **covariance** C1, C2 (64×64, "how channels co-oscillate"). Solve the **generalized eigenproblem `eigh(C1, C1+C2)`**: it returns directions `w` that maximize the ratio `wᵀC1w / wᵀ(C1+C2)w`, each with eigenvalue **λ = a/(a+b)** = the fraction of that filter's oscillation belonging to class-1. λ→1 = class-1 detector, λ→0 = class-0, **λ=0.5 = useless**. We keep the **4 filters with λ farthest from 0.5**. (Measured: filters split as λ = 0.378 and 0.619, and `a/(a+b)` equals λ exactly.)

**(4) Feature = log-variance.** Per virtual channel: variance → normalize to a fraction → **log**. Log because variance is right-skewed (log → Gaussian-ish, good for LDA) and turns multiplicative differences into additive ones (good for a linear boundary). Measured: with log 0.896 vs without 0.884.

**Glossary (one thing, several names):** *recipe = spatial filter = eigenvector = a row of `filters_` = 64 weights* ↔ *score = λ = eigenvalue = a/(a+b)*. Leading `_` (e.g. `_class_cov`) = private helper; trailing `_` (e.g. `filters_`) = learned during `fit` (sklearn convention).
**Why CSP not PCA?** PCA is unsupervised (largest variance, fooled by big artifacts); CSP is **supervised** (largest *difference between classes*). Our task is discrimination → CSP.

### 2.8 LDA = drawing a fence (`tpv/own_lda.py` for the bonus)
The 4 CSP numbers = a **point in 4-D**; the two classes form two clouds. There are **two lines**: the **direction `w`** (an arrow *through* the clouds, center-to-center) and the **fence** (perpendicular to `w`, *between* the clouds). We compute `w`; the fence follows.
Math: centers `μ0, μ1`; **`w = Σ⁻¹(μ1 − μ0)`** (the center-to-center direction corrected by the spread `Σ`, so noisy axes are trusted less); decision = sign of **`w·x + b`**. LDA also reduces 4→1 (project onto `w`) but then thresholds to output a class — so it's a *classifier*, not just a reducer.
Measured (exp2): μ0=[-1.93,-1.78,-1.07,-1.13], μ1=[-1.11,-1.22,-1.79,-1.77], w=[12.92,7.86,-10.4,-2.13], b=13.46 → all "hands" negative, all "feet" positive = 100% separated.

### 2.9 Cross-validation, leakage, and the A/B split
**Cheating:** scoring on the trials you trained on inflates the result (measured: 1.0).
**`cross_val_score`** = repeat "hide 20 % (9 trials), train on 80 % (36), score on the hidden 9" **10 times** (`ShuffleSplit(10, test_size=0.2)`) and average. It's literally a loop that **builds a fresh model each round and discards it**; the manual loop reproduces it exactly. Measured (subj 4, exp 3): folds `[0.44,1,0.56,…]`, mean **0.7333** (each fold is k/9).
**Leakage:** CSP also *learns*, so if CSP sees the test trials while computing filters, that's hidden cheating. Fix: put CSP **first in the Pipeline** so `cross_val_score` **refits CSP per fold on training data only**. Measured: leaky 1.0 vs correct 0.844 (a **0.156** inflation).
**Two independent 80/20 splits in `train`:** **(A)** `ShuffleSplit×10` over all data → the **honest score** (this is what the 60 % gate uses); **(B)** one `train_test_split` → fit the **deliverable model** on 80 %, save it + the held-out 20 % for `predict`. A's 10 models are thrown away; B is the model you ship. A exists because a single split is luck-dependent, so we average 10.

### 2.10 sklearn integration = a "standard plug"
sklearn = a toolbox with standard parts and a standard wiring spec; **Pipeline** = an assembly line (CSP stage → LDA stage). **Inheritance** = a child class gets the parent's abilities for free. `class MyCSP(TransformerMixin, BaseEstimator)`:
- **BaseEstimator** → `get_params`/`set_params` → sklearn can **clone, tune, inspect** it.
- **TransformerMixin** → `fit_transform` for free → it's a recognized **transformer**.
Because of this it plugs into the Pipeline socket and `cross_val_score` can **clone + refit it per fold** — which is precisely what makes leakage-free scoring possible. Not bureaucracy: it's the foundation of honest evaluation.

### 2.11 Subject-specific models (not one universal net)
Each model is fit **per subject** — it answers "*which action*", not "which person". EEG differs per brain/cap, so per-subject (calibration) models are the BCI standard and usually beat a generic one. Training is **fast and analytic** (one eigendecomposition + one closed-form LDA), unlike a deep net trained for hours. The 60 % gate trains **one model per subject** and averages.

---

# 3. Evaluation sheet — item by item

## 3.1 Preprocessing — *"Watch it for the plot"*
**📋 Criterion:** data is parsed and **visualized by a script showing raw vs filtered**, the filtered signal being **cleaner**.

**🗣️ Say:** "One script shows the raw signal, the 7–30 Hz filtered signal, and the PSD (power per frequency), before and after."
**🖥️ Show:** `python scripts/visualize.py 4 14` *(any subject/run works — offer to let the examiner pick a number to prove it's not cherry-picked).*
**👉 Time series:** "Raw has large slow baseline drift; filtered sits around zero and is smooth; the shaded bands are annotations (blue=T0 rest, orange=T1, green=T2) and the task-related activity is preserved."
**👉 PSD:** "Before: power piled up at low frequencies plus a sharp **60 Hz line-noise spike**, content out to 80 Hz. After: only **7–30 Hz survives**, a steep drop past 30 Hz, and the 60 Hz spike is gone — that is the direct evidence of 'cleaner'."

**❓ Q&A**
- *Why 7–30 Hz?* — Motor-imagery mu/beta rhythms (ERD) live there; the sheet itself asks for ~8–40 Hz.
- *What filter type?* — A **FIR** designed with **firwin**; **zero-phase**, so it doesn't shift event timing.
- *Why does the filtered cutoff slope instead of dropping vertically?* — A finite FIR has a steep but finite transition band; frequencies just outside 7/30 Hz are attenuated progressively, not zeroed.
- *Why average reference?* — Removes activity common to all electrodes (reference choice, global noise), isolating each channel's local activity.
- *Which video?* — The subject and submission contain no video; the operative requirement is "filtered is cleaner", which the PSD demonstrates unambiguously.
- *What are the shaded colors?* — MNE auto-assigns them; here blue=T0, orange=T1, green=T2 (read the legend, the colors carry no meaning).

→ **Yes**

## 3.2 Feature extraction
**📋 Criterion:** the filtering must **mean something** — the **significant motor-imagery frequencies (~8–40 Hz) are kept**. (Learning to select relevant frequencies = bonus.)

**🗣️ Say:** "The core check is that I keep the meaningful band. My pass-band is **7–30 Hz**, preserving mu (8–12) and beta (13–30) — the bands where ERD lives. Keeping an arbitrary band (say 50–60 Hz) would retain a signal but be meaningless for motor imagery; 7–30 Hz is meaningful *in the context of the data*."
**🖥️ Show:** `FMIN = 7.0, FMAX = 30.0` (from `config.py`); plus the filtered PSD from 3.1.
> "The 64×321→4 *compression* itself (CSP) is scored under **Implementation**; the 'learn which frequencies' part is my **FBCSP** bonus, demoed in the bonus section."

**❓ Q&A**
- *Why stop at 30, not 40?* — ERD is strongest in 8–30 Hz; above 30 Hz the noise-to-signal rises. 8–30 Hz is a standard BCI choice, validated by the 0.658 result.
- *What features actually feed the classifier?* — Not the raw filtered signal, but **CSP log-variance** (4 numbers per trial), i.e. the ERD power in the most discriminative spatial patterns.

→ **Yes**

## 3.3 Train
**📋 Criterion:** a **train mode**; **sklearn validation tools** used; the **training score displayed**.

**🗣️ Say:** "`train` mode runs `cross_val_score`, prints the score, then saves a model."
**🖥️ Show:** `python mybci.py 4 14 train`
```
[0.4444 1.0000 0.5556 0.6667 0.6667 1.0000 1.0000 0.6667 0.6667 0.6667]
cross_val_score: 0.7333
```
**👉 Say:** "The array is the accuracy of **10 cross-validation folds**; 0.7333 is their mean — the training score. The validation tool is `cross_val_score` with `ShuffleSplit(10, test_size=0.2)`: each fold trains on 36 trials, tests on a hidden 9, so values are k/9 and the average smooths luck. Then I fit on 80 % and save the model + the held-out 20 %."

**❓ Q&A**
- *Which validation tool exactly?* — `cross_val_score` over the **whole pipeline**, `ShuffleSplit(n_splits=10, test_size=0.2)`.
- *Is there leakage?* — No: the pipeline (CSP **and** LDA) is refit each fold on training data only, because CSP is the first pipeline step.
- *Does `cross_val_score` train once and test at the end?* — No: it builds a **fresh model each of the 10 rounds**, scores, and discards it. The saved model is a separate fit on 80 %.
- *0.73 here but the gate is 60 %?* — The 60 % applies to the **mean over 109 subjects × 6 experiments (0.658)**, not to one subject.
- *Reproducible?* — Default splits vary run-to-run; `TPV_SEED=42` fixes them.

→ **Yes**

## 3.4 Predict
**📋 Criterion:** a **predict mode**, also using validation tools; the **prediction output is displayed** (class id suffices).

**🗣️ Say:** "`predict` loads the saved model and replays the held-out 20 % (9 trials) one at a time, comparing prediction to truth — no retraining."
**🖥️ Show:** `python mybci.py 4 14 predict`
```
epoch 00:  [2]  [2] True
...
Accuracy: 0.6667
```
**👉 Say:** "Each line is **[prediction] [truth] match**; ids are 1=imagined hands (T1), 2=imagined feet (T2). These 9 trials were **never seen during training**, so this is an honest prediction; 6/9 = 0.6667."

**❓ Q&A**
- *Where are the 'validation tools'?* — Prediction runs on the **held-out validation set** and accuracy is computed against ground truth.
- *Why only 9 trials?* — They are the 20 % `train` set aside for this subject/experiment.
- *Why 0.6667 here vs 0.7333 in train?* — train's number is a 10-fold average; this is one fixed 9-trial test. Small sample, different estimator.
- *Does predict retrain?* — No, it only loads the `.joblib`. Inference is CSP transform + one LDA dot product.
- *Must predict use the same subject/run as train?* — Yes: the artifact is named per subject/run; predict raises a clear error telling you to train first. Models are subject-specific.

→ **Yes**

## 3.5 Realtime
**📋 Criterion:** prediction made **as data is streamed**; result output **between 0 and 2 seconds** after the event.

**🗣️ Say:** "`predict` feeds the pipeline **one event (2-second epoch) at a time**, like a stream. Measured latency per event is ~0.1–0.5 **milliseconds**, and a `assert latency < 2.0` enforces the budget in code."
**🖥️ Show:** per-event latency `0.0001–0.0005 s  OK (<2s)`.
**🗣️ The '2 seconds':** "After the event you collect a 2-second window (TMIN=0…TMAX=2.0) to form the epoch; inference is sub-millisecond, so the result is available within 2 seconds."
**🗣️ Be upfront:** "The data is pre-recorded, so this is a **simulated stream** — I replay held-out epochs one by one rather than using live hardware. I implemented it with a loop instead of `mne-realtime`, which satisfies 'streamed, within 2 s' without an external real-time dependency."

**❓ Q&A**
- *Is it a real live stream?* — No, a per-event simulation over pre-recorded data; the processing model (one event at a time, 2 s budget) is identical to live.
- *Why so fast?* — Inference is a matrix multiply + variance + a dot product; no iterative computation.
- *Where is 2 s guaranteed?* — `assert latency < LATENCY_BUDGET_S` in `predict.py`, and the epoch itself is the 2 s post-event window.

→ **Yes**

## 3.6 Integration
**📋 Criterion:** integrated into the **sklearn pipeline**, inheriting **`BaseEstimator`** and **`TransformerMixin`**.

**🗣️ Say:** "`class MyCSP(TransformerMixin, BaseEstimator)` gives it the transformer contract (`fit`/`transform`), so it drops into `Pipeline([('csp', MyCSP()), ('clf', LDA())])`."
**🖥️ Show:** `isinstance(csp, BaseEstimator)=True`, `isinstance(csp, TransformerMixin)=True`, `get_params()={...}`, has `fit_transform`, `clone()` works.
**👉 Say:** "BaseEstimator gives get/set_params (clone, tuning); TransformerMixin gives fit_transform. That's what lets `cross_val_score` **clone and refit per fold** — so this inheritance is the basis of leakage-free scoring, not a formality."

**❓ Q&A**
- *Why both base classes?* — BaseEstimator = parameter management (clone/tune/inspect); TransformerMixin = transformer contract. Both are needed to be a full sklearn component.
- *Why does `__init__` do no computation?* — sklearn contract: `__init__` only stores hyperparameters, learning happens in `fit`; otherwise `clone` breaks.
- *Did you avoid the library CSP?* — Yes; `mne.decoding.CSP` is used only as a parity reference in tests.

→ **Yes**

## 3.7 Implementation — dimensionality reduction (high-value)
**📋 Criterion:** a **dimensionality-reduction algorithm is implemented** (PCA/CSP or other) and the student shows **general understanding**. numpy/scipy are allowed for eigendecomposition / SVD / covariance.

**🗣️ Say + 🖥️ Show:** "I implemented **CSP from scratch** in `csp.py`. Imports are only `numpy`, `scipy.linalg.eigh` (explicitly allowed), `sklearn.base`, and my own `generalized_eigh` — **no library CSP**."

**🗣️ Explain the algorithm (Part 2.7 in brief):** (1) reduce because the answer is oscillation size and 45 trials can't fill 20k-D; (2) per-class covariance C1, C2; (3) `eigh(C1, C1+C2)` → filters whose λ = a/(a+b) is farthest from 0.5; (4) blend with `@`, take log-variance → 4 numbers.

**🖥️ Validate:** `pytest tests/test_csp_parity.py -v` → **PASSED** — my CSP matches `mne.decoding.CSP` within 0.05 on the same splits (exp3), and ≥ 0.60 on a single subject.

**🗣️ Code tour (`csp.py`, for deep questions):**
- `__init__` (13–17): stores hyperparameters only.
- `_class_cov` (19–35): `E@E.T` (per-trial covariance) → **trace-normalize** (so a loud trial doesn't dominate) → average → symmetrize (float round-off) → **shrinkage** (`reg=0.01`, blend toward diagonal so `eigh` stays well-conditioned).
- `fit` (37–66): requires `y` (supervised) → C1,C2 → `eigh(C1,C1+C2)` → sort by `|λ-0.5|` desc → `filters_` (top 4×64); `patterns_` (pseudo-inverse, for visualization).
- `transform` (68–78): `filters_ @ E` → variance → normalize → log → 4 numbers.

**❓ Q&A**
- *What is an eigenvalue here?* — The fraction of a filter's oscillation belonging to one class; far from 0.5 = discriminative.
- *What is covariance here?* — A 64×64 table of how channels co-oscillate within a class; the raw material for CSP.
- *Why a generalized eigenproblem (two matrices)?* — Because we maximize the **ratio** `wᵀC1w/wᵀ(C1+C2)w`, which puts (C1+C2) in the denominator.
- *Why CSP over PCA?* — PCA is unsupervised (largest variance, fooled by artifacts); CSP is supervised (largest class difference).
- *Why log-variance, not variance?* — De-skews toward Gaussian and linearizes multiplicative differences for the linear classifier.
- *`filters_` vs `patterns_`?* — filters_ maps signal→feature (analysis); patterns_ maps feature→scalp (interpretation/topomap).
- *Difference from MNE's CSP?* — Same algorithm; mine matches within 0.05 accuracy and ~1e-14 on the eigenproblem (bonus solver).

→ **Yes**

## 3.8 Score (the 60 % gate)
**📋 Criterion:** a script trains over **each subject**, averages **by experiment type** → the **mean of the six means ≥ 60 %**. Above 60 %, +1 point per 1 % (0–5).

**🗣️ Say:** "`python mybci.py` (no args) cross-validates all 109 subjects × 6 experiments. For each experiment it averages over subjects (6 means), then averages those 6."
**🖥️ Show:** `python scripts/validate_60.py` (run the **full** version at the defense; `--fast 5` is only a preview).
```
experiment 0..5:  per-experiment means
Mean accuracy of 6 experiments:  0.658
```
**🗣️ Say:** "Full 109-subject result is **0.658 ≥ 0.60 → pass**. Grading: 65.8 % is 5.8 points over 60 → capped at **5/5**."

**❓ Q&A**
- *How is 60 % computed?* — Mean per experiment (6 numbers), then the mean of those 6 — not one flat average.
- *Why does `--fast 5` show ~0.71 but full is 0.658?* — The first 5 subjects happen to be easy; the full 109 includes harder ones. The official number is the full run.
- *Skipped subjects?* — `run_all` skips **only** data/IO errors (narrow catch); genuine algorithm bugs propagate, never hidden.
- *How long?* — ~2.5 min once cached; first run also downloads ~3.1 GB.

→ **Yes / 5/5**

## 3.9 Bonus · Datasets
**📋 Criterion:** other datasets processed? scoring correct, accounting for noise/quality vs the subject dataset?

**🗣️ Say:** "I added **BCI Competition IV-2a** via moabb — a structurally different dataset (**22 channels, 250 Hz, 288 trials**), yet my pipeline runs **unchanged**."
**🖥️ Show:** `python scripts/bonus_demo.py` → `[G] 2nd dataset BCI IV-2a (subj 1) : 0.8552` (shape 288×22×1001).
**🗣️ Say:** "It works unchanged because **MyCSP is channel-count-agnostic** (covariance is n_ch × n_ch), and `external.py` passes my 7–30 Hz band to moabb. **0.855 > 0.658 because IV-2a is a curated competition dataset (less noise)** — exactly what the criterion's quality caveat predicts. It downloads into the project's `mne_data/` and is a separate run from the mandatory gate."

**❓ Q&A**
- *Why does 64-ch code run on 22-ch data?* — No channel-count assumption anywhere; covariance and the eigenproblem size to the input.
- *Is 0.855 > 0.658 suspicious?* — No: cleaner data → higher score; the criterion asks to weigh quality.
- *Same preprocessing?* — Yes, 7–30 Hz via the moabb paradigm; same left-vs-right-hand imagery task.

→ **Yes**

## 3.10 Bonus · Feature engineering
**📋 Criterion:** relevance of preprocessing and how data is fed; **Fourier/wavelet or any pre-processing transform is a plus**.

**🗣️ Say:** "Two layers of frequency transformation. (1) The mandatory **FIR band-pass** is already a frequency-domain transform. (2) The bonus **Filter-Bank CSP (FBCSP)** splits 7–30 Hz into **four sub-bands (8-12 / 12-16 / 16-20 / 20-30)** and runs a separate CSP per band → 8 features."
**🗣️ Why split (depth):** "A single 7–30 Hz band lumps mu, low-beta and high-beta together, but **ERD strength and topography differ by frequency** — some subjects are mu-dominant, others beta-dominant. FBCSP learns a **separate spatial pattern per band** and lets the classifier weight which bands matter — a frequency-resolution that plain CSP lacks. It's in the Fourier/wavelet family of frequency-domain feature engineering."
**🗣️ How data is fed:** "Each sub-band → its own CSP → log-variance features → **concatenated** into the classifier. The band-pass lives inside `fit`/`transform`, so `cross_val_score` refits per fold (no leakage)."
**🖥️ Show:** `[F] Filter-Bank CSP (4 sub-bands) : 0.8222` (plain CSP 0.8444).
**🗣️ Be honest:** "Here 0.822 < 0.844 — eight features on a small dataset can mildly overfit. FBCSP is strictly **more expressive** and can win on other subjects/datasets; this demonstrates a more sophisticated feature-engineering method."

**❓ Q&A**
- *What does FBCSP add over CSP?* — Per-sub-band spatial patterns, so the classifier learns band importance.
- *Why isn't it always better?* — More features → mild overfitting on ~45 trials; it's a sophistication demo, not a guaranteed win.

→ **Yes**

## 3.11 Bonus · Implementations
**📋 Criterion:** how deep — own **eigendecomposition / SVD / covariance**? a **complex** dimensionality reduction? **hyperparameter tuning/learning**? own **classifier**?

### A. From-scratch eigensolver (own eigendecomposition)
**🗣️ Say:** "Even `eigh` is hand-written — a **pure-numpy cyclic Jacobi** solver in `jacobi.py`."
**🗣️ How it works (depth):** "It zeroes off-diagonal entries one pair at a time via rotations, sweeping until the matrix is diagonal — the diagonal is the eigenvalues, the accumulated rotations are the eigenvectors. The **generalized** problem `C1 w = λ(C1+C2)w` is reduced to standard form by **whitening**, then Jacobi runs twice. It uses only `+,*,sqrt,sign,matmul` — no `np.linalg.eig/eigh/svd`, no scipy."
**🖥️ Show:**
```
[A] Jacobi accuracy            : 0.8444   (identical to scipy eigh)
    standard eig vs numpy      : 5.33e-14   ← machine precision
    generalized eig vs scipy   : 1.55e-15
```
Enable with `MyCSP(solver="jacobi")`.

### C. Hyperparameter tuning (tuning/learning)
**🗣️ Say:** "`GridSearchCV` searches `n_components ∈ {4,6,8}` in `evaluate.tune`, refitting CSP inside each inner fold — **leakage-free nested CV**. The official gate stays fixed at 4 for fair comparison; tuning is a demo."
**🖥️ Show:** `[C] tuned {'csp__n_components': 8} -> cv 0.8444`.

### D. From-scratch classifier (own classifier)
**🗣️ Say:** "LDA is also hand-written in `own_lda.py`: pooled within-class covariance, **`w = Σ⁻¹(μ1−μ0)`**, prior-corrected intercept, decision = sign(`w·x+b`), using only `np.linalg.solve`."
**🖥️ Show:**
```
[D] OwnLDA accuracy        : 0.7889
    agreement w/ sklearn LDA: 100.0%
```
Enable with `build_pipeline(clf="own-lda")`.

(+ **F. FBCSP** also counts as the "complex dimensionality reduction" item.)

**❓ Q&A**
- *Really no eigendecomposition library?* — `jacobi.py` uses only basic ops; it matches scipy to ~5e-14.
- *How is the generalized problem solved?* — Whitening to standard form, then Jacobi twice.
- *Is the tuning leakage-free?* — Yes; CSP is refit inside each inner fold.
- *Does OwnLDA match sklearn?* — 100 % identical predictions (same pooled covariance + Σ⁻¹(μ1−μ0)).

→ **Yes**

## 3.12 Ratings
**🗣️ Say:** "None of the negative flags apply (Empty / Incomplete / Cheat / Crash / **Forbidden function**). In particular, the sheet explicitly allows numpy/scipy for eigendecomposition, SVD and covariance — and I implemented even those myself. Mandatory passes at 0.658 and all five bonuses are implemented and verified, so I'd argue **Outstanding project**."
→ **Outstanding**

## 3.13 Conclusion
The examiner writes the comment. Closing line to offer:
> "PhysioNet's 64 channels are band-passed to 7–30 Hz to expose ERD, compressed by a **from-scratch CSP** into 4 numbers, classified by LDA, and scored with **leakage-free cross-validation** to a 109-subject × 6-experiment mean of **0.658 (≥ 60 %)**; all five bonuses — including a from-scratch eigensolver — are implemented."

---

# 4. Command runbook (in order)

```bash
source .venv/bin/activate

python scripts/validate_60.py              # full 60% gate → 0.658  (run BEFORE defense too)
pytest -m "not network"                    # fast unit tests
pytest tests/test_csp_parity.py -v         # scratch CSP vs mne, within 0.05
python scripts/visualize.py 4 14           # raw vs 7–30Hz + PSD
python mybci.py 4 14 train                 # cross_val_score + save model
python mybci.py 4 14 predict               # stream held-out trials
python mybci.py                            # the 6-experiment means → 0.658
python scripts/bonus_demo.py               # bonuses A,C,D,F,G in one shot
```

---

# 5. Final checklist

| Sheet item | Status | Evidence (measured) |
|---|---|---|
| Preprocessing (plot) | ✅ | raw vs 7–30 Hz + PSD; 60 Hz/drift removed |
| Feature extraction | ✅ | mu/beta (7–30 Hz) kept |
| Train | ✅ | `cross_val_score` 0.7333 shown |
| Predict | ✅ | held-out 6/9 = 0.667 printed |
| Realtime | ✅ | ~0.0005 s/event < 2 s |
| Integration | ✅ | BaseEstimator + TransformerMixin; clone works |
| Implementation (dim. reduction) | ✅ | from-scratch CSP; parity test PASS (<0.05) |
| Score | ✅ **5/5** | 0.658 ≥ 0.60 |
| Bonus: Datasets | ✅ | BCI IV-2a 0.855 (22ch/250Hz) |
| Bonus: Feature engineering | ✅ | FBCSP 0.822 (4 sub-bands) |
| Bonus: Implementations | ✅ | Jacobi 5e-14 · tuning · OwnLDA 100 % |
| Ratings | ⭐ Outstanding | no negative flags |

---

# 6. Cross-cutting hard-question bank (curveballs)

**On method & design**
- *Why is exp0 (left vs right hand) your worst experiment (~0.57)?* — Both are hands, on adjacent, near-symmetric motor cortex, so the contralateral ERD difference is small; hands-vs-feet (distinct, midline-vs-lateral) is much easier (~0.71–0.84).
- *Your accuracies vary a lot per subject — is that a bug?* — No, BCI is famously subject-variable ("BCI illiteracy"); that's why the gate is a 109-subject mean, and why I average 10 folds per subject.
- *Would a deep network beat this?* — Possibly with far more data and per-subject calibration, but classic CSP+LDA is the interpretable, fast, data-efficient standard and meets the bar; the subject also asks specifically for a dimensionality-reduction algorithm.
- *What's the single biggest risk to your score?* — Non-discriminative or noisy subjects; mitigated by shrinkage-regularized covariance and 10-fold averaging.

**On correctness & rigor**
- *Prove there's no data leakage.* — CSP is the first Pipeline step, so `cross_val_score` clones and refits the whole pipeline per fold; I can show the leaky-vs-correct gap (1.0 vs 0.844). FBCSP's band-pass is also inside fit/transform for the same reason.
- *Prove your CSP is correct, not just plausible.* — Parity test vs `mne.decoding.CSP` (<0.05 accuracy on identical splits); the bonus Jacobi solver matches scipy to ~1e-14.
- *Why trace-normalize the covariance?* — So a single high-amplitude trial doesn't dominate the class average; it makes trials comparable.
- *Why the `reg` shrinkage?* — Keeps the covariance well-conditioned so `eigh` is numerically stable on few/colinear channels.
- *Is your 0.658 reproducible?* — `TPV_SEED=42` fixes splits; unseeded it fluctuates slightly around 0.65 but the gate is on the grand mean.

**On the libraries / "forbidden function"**
- *Did you use a forbidden function?* — No. The sheet allows numpy/scipy for eigendecomposition/SVD/covariance; I use scipy `eigh` and numpy covariance, and additionally re-implemented both from scratch (Jacobi, OwnLDA).
- *Why use `mne` and `scikit-learn` at all?* — The subject mandates them: MNE for EEG I/O/filtering/epoching, sklearn for the pipeline and validation. The *algorithm* (CSP) is mine.

**On the data**
- *Where is the data stored?* — In the project's `./mne_data/` (PhysioNet ~3.1 GB and BCI IV-2a ~83 MB), gitignored, so it travels with the folder but isn't committed.
- *What is one epoch's shape and why?* — (64, 321): 64 channels × (2 s × 160 Hz + 1) samples, the +1 from inclusive endpoints.
- *What do T1/T2 mean?* — They're the dataset's action annotations; their meaning depends on the run group (e.g., runs 3/7/11 → left/right fist; 5/9/13 → both fists/both feet).

**If something fails live**
- *A subject errors out.* — `run_all` skips only data/IO failures and reports them; the grand mean is over the rest. I can rerun that subject's `train` to show it in isolation.
- *No internet for the bonus dataset.* — It's pre-downloaded into `./mne_data/`; if missing it's an 83 MB fetch on first `bonus_demo` run.

> **You are ready.** Internalize Part 2, walk Part 3 in order, and keep Part 6 in your back pocket.
