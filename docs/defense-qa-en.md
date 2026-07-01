# Total Perspective Vortex — Defense Q&A (English)

> A defense-ready walkthrough organized around the **official evaluation sheet**.
> **Every concept lives inside the evaluation item that needs it** — so you never jump to a separate "concepts" chapter. For each item you get: **📋 the verbatim criterion → 📚 the concept you need → 🗣️ what to say / 🖥️ what to show → ❓ a deep Q&A bank** (routine *and* tough examiner questions).
> The evaluation is conducted in English; every answer is phrased so you can say it almost verbatim. Numbers are measured from this codebase. Deeper math: [`defense-guide.md`](defense-guide.md); diagrams: [`workflows.md`](workflows.md).

Legend: 📋 criterion · 📚 concept · 🗣️ say · 🖥️ run/show · 👉 point at screen · ❓ question + answer.

---

# 0. Before the defense (do once)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fetch datasets into the PROJECT folder + confirm the 60% gate (slow the first time).
python scripts/validate_60.py            # full 109 subjects → grand mean 0.658
python scripts/bonus_demo.py             # also pulls the bonus dataset (BCI IV-2a) into ./mne_data
```
Both datasets live inside the project's `./mne_data/` (PhysioNet ~3.1 GB, BCI IV-2a ~83 MB), gitignored. On the defense machine, run those two once beforehand (internet needed for the first download).

---

# 1. Sixty-second pitch + project map

**Pitch:** "This is an EEG brain–computer interface. From PhysioNet motor-movement/imagery recordings (109 subjects, 64 channels) I classify which action a subject performed or imagined — left vs right hand, hands vs feet, real vs imagined — from **only the brain signal**. Three stages: **preprocessing** (average reference, 7–30 Hz band-pass, 2-second epoching), **dimensionality reduction** (a from-scratch **CSP** turning 64 channels into 4 discriminative numbers), and **classification** (LDA). I score it honestly with `cross_val_score`. The mandatory bar is a from-scratch reducer integrated into sklearn and a **mean accuracy ≥ 60 %** across the six experiment types over all 109 subjects — I reach **0.658** — plus all five bonuses."

**Map:** `load EDF → filter 7–30 Hz → epoch (2 s) → CSP (64×321→4) → LDA → cross-val → 60% gate`.

---

# 2. Walkthrough — evaluation sheet, item by item (concepts included)

## 2.0 Setup & "what is this project"
**🗣️ Say:** "This is my submission directory — a Python project: entry point `mybci.py`, core package `tpv/`, scripts in `scripts/`. Setup is a venv + `pip install -r requirements.txt`; the dataset (~3.1 GB) auto-downloads into the project's `mne_data/` on first run."
**🖥️ Show:** `ls` → `source .venv/bin/activate` → (optional) `pytest -m "not network"` (all pass).
Then give the 60-second pitch above.

---

## 2.1 Preprocessing — *"Watch it for the plot"*
**📋 Criterion (verbatim, evaluation sheet):**
> *"Check if the data were parsed then visualized with a script, showing raw and filtered data. The plots should look like what is shown in the video, the filtered signal being "cleaner"."*

**📚 Concept — what you need to know:**
- **EEG = 64 microphones on the scalp.** 64 electrodes sample voltage **160×/s** in **µV** → a `(64, 20000)` table per ~125 s recording. The plot shows this signal **before vs after filtering**.
- **The filter (this is what the plot is about).** ① **Average reference**: at each instant subtract the **mean of all 64 channels** (removes what's common to all electrodes). ② **7–30 Hz band-pass (FIR, firwin, zero-phase)**: keep the motor-imagery band, drop slow drift (<7 Hz) and muscle/line noise (>30 Hz). *Why 7–30 Hz is the meaningful band → Feature extraction.* We filter the long continuous signal first, then epoch.
- **Then epoching (formatting — not shown in this plot).** Annotations mark **T0 (rest), T1/T2 (the two actions)**; we drop rest and cut a **2-second window** at each T1/T2 → one trial `(64 × 321)` (321 = 2 s × 160 + 1; ~45 trials/subject). The raw-vs-filtered plot shows the **continuous** signal, not epochs.

**🗣️ Say:** "Let me start with preprocessing — this is where the raw recording becomes usable. One script loads a subject's EEG and shows it in **two complementary views, each before and after filtering**: first the **time series** (raw wavy signal vs the 7–30 Hz filtered one) so you see the cleanup with your eyes, then the **PSD (power spectral density)** — an equalizer-style view of how much energy sits at each frequency. The two views make the same point from different angles: the filtered signal is genuinely *cleaner*, because I strip the slow drift and line noise and keep only the motor band (mu/beta) that carries the answer."
**🖥️ Show:** `python scripts/visualize.py 4 14` *(any subject/run works — offer to let the examiner pick a number).*
**👉 Time series:** raw has big slow drift; filtered is smooth around zero; shaded bands = annotations (blue=T0, orange=T1, green=T2); task activity is preserved.
**👉 PSD:** before = power across all 0–80 Hz + a sharp **60 Hz line-noise spike**; after = only 7–30 Hz survives, steep drop past 30 Hz, spike gone — the direct proof of "cleaner".

**❓ Q&A**
- *What are the two kinds of plot, and why both?* — Same signal, two views. Time series lets you *see* it get smoother; PSD *proves* which frequencies survived. Filtering is one process; these are two views of its result.
- *What did the filter actually do?* — The raw signal mixes frequencies ~0–80 Hz (80 = half the 160 Hz sampling rate, the Nyquist limit). The band-pass **keeps 7–30 Hz and pushes everything below 7 and above 30 to near-zero.**
- *Both PSDs span 0–80 Hz but look different — why?* — Same x-axis on purpose; only the **content** differs (before vs after). Showing the full range is what lets you see the out-of-band power was removed.
- *PSD axes?* — x = frequency (Hz); y = power in **dB** (log scale), how strong each frequency is.
- *Why negative dB?* — dB is **relative to a reference (1 µV², the `re 1 µV²` in the label)**, like floors relative to a lobby. Negative = weaker than the reference, not negative power.
- *Why 7–30 Hz, not the sheet's 40?* — mu/beta ERD is strongest in 8–30 Hz; above 30 the noise rises. 8–30 is a standard BCI choice, validated by 0.658, and covers the requested core.
- *Filter type?* — FIR via firwin, **zero-phase** (no timing shift); a steep slope, not a vertical cliff (finite FIR).
- *What exactly is average reference?* — Subtract the per-timepoint mean of all 64 channels from each channel.
- *Doesn't that average include each channel's own signal?* — It does, but the **common part survives averaging while the unique parts (different signs) cancel**, so the average ≈ common; subtracting it costs each channel only ~1/64 (~1.5 %) of its own signal.
- *Two different "references" — don't confuse:* average reference *changes the signal* (`preprocessing.py`); the 1 µV² reference is only the dB yardstick on the PSD plot (MNE display, `viz.py`).
- *Where is the PSD computed?* — `viz.py` via MNE's `compute_psd(method="welch").plot()`; it's a visualization, **not** fed to the model — the pipeline uses the filtered *time series* → epochs → CSP.
- *One epoch's shape, and why?* — `(64, 321)`: 64 channels × (2 s × 160 Hz + 1) samples, +1 from inclusive endpoints.
- *What do T1/T2 mean?* — Dataset action annotations; meaning depends on the run group (runs 3/7/11 → left/right fist; 5/9/13 → both fists/both feet).
- *Which video?* — The subject/submission contain no video; the operative requirement is "filtered is cleaner", which the PSD shows.

→ **Yes**

---

## 2.2 Feature extraction
**📋 Criterion (verbatim, evaluation sheet):**
> *"Its nice to filter a signal, but it needs to mean something in the context of your data. Check that the significative frequencies for a motor imagery task are kept (~8-40Hz). If the program learns to select the relevant frequencies for classification its better, cf bonus questions."*

**📚 Concept — what you need to know:**
- This item asks only whether the filtering is **meaningful, not arbitrary** — does it keep the frequencies that actually carry motor-imagery information? (Not about the CSP compression — that's Implementation.)
- **Why mu/beta (7–30 Hz) is the meaningful band:** the motor cortex has two characteristic rhythms — **mu (8–12 Hz)**, its idle "hum" at rest, and **beta (13–30 Hz)**, its active-control rhythm — both produced by neurons firing *in sync*. The instant you **move or imagine** moving a limb, that region gets busy, the synchrony breaks, and the mu/beta oscillation **shrinks** (= **ERD**), **contralaterally** (right hand → drop over the left motor cortex, C3). So the *size* of the mu/beta rhythm is a direct readout of "is this motor area engaged, and which side?" — exactly the information we classify.
- **Why nothing else:** below 7 Hz is slow drift / eye movement; above 30 Hz is muscle and 50/60 Hz line noise — neither carries motor-imagery information. The signal is concentrated in 8–30 Hz, so **7–30 Hz keeps it and drops the rest** — that's what makes the filtering meaningful.

**🗣️ Say:** "This item checks whether my filtering *means something*, not just any filter. It does: I keep **7–30 Hz**, exactly the mu and beta bands where motor-imagery ERD lives. A filter that kept, say, 50–60 Hz would look clean but carry no motor information. The PSD already showed mu/beta preserved and the rest removed. (Learning which sub-frequencies matter is the FBCSP bonus.)"
**🖥️ Show:** `FMIN = 7.0, FMAX = 30.0` (from `config.py`) + the filtered PSD showing mu/beta retained.

**❓ Q&A**
- *Why is 7–30 Hz "meaningful" and not arbitrary?* — mu (8–12) and beta (13–30) are the rhythms that change with motor imagery (ERD); other bands don't carry that information.
- *Why stop at 30, not the sheet's 40?* — ERD is strongest in 8–30 Hz; above 30 the noise rises. 8–30 is a standard BCI choice, validated by 0.658, and covers the requested core.
- *What would a "meaningless" filter look like?* — Keeping a band with no motor information (e.g. the 50–60 Hz line-noise region), or not filtering at all.
- *Where does "learn the relevant frequencies" come in?* — The FBCSP bonus (item 2.10) splits the band into sub-bands and lets the classifier weight which matter.

→ **Yes**

---

## 2.3 Train
**📋 Criterion (verbatim, evaluation sheet):**
> *"The program has a train mode, sklearn score validation tools are used. The score for the training is displayed."*

**📚 Concept — what you need to know:**
- **Why not just score on the training trials?** That's cheating — the model memorizes (measured: 1.0).
- **Cross-validation (`cross_val_score`)**: repeat "hide 20 % (9 trials), train on 80 % (36), score the hidden 9" **10 times** (`ShuffleSplit(10, test_size=0.2)`) and average. A **fold** = one such round. The numbers: 45 = this subject's trials, 9 = 20 %, 36 = 80 %, 10 = `n_splits`; each fold score is k/9.
- It's literally a loop that **builds a fresh model each round and discards it** — not one training run.
- **Leakage**: CSP also *learns*, so it must be refit per fold on training data only. Putting CSP **first in the Pipeline** achieves this (leaky vs correct: 1.0 vs 0.844).
- **Two independent 80/20 splits in `train`:** **(A)** `ShuffleSplit×10` → the honest score (what the gate uses); **(B)** one `train_test_split` → fit a **deployable model** on 80 % and save it + the held-out 20 % for `predict`. A's 10 models are thrown away.

**🗣️ Say:** "`train` runs `cross_val_score` over the whole pipeline, prints the fold scores and their mean, then fits and saves a model."
**🖥️ Show:** `python mybci.py 4 14 train`
```
[0.4444 1.0000 0.5556 0.6667 0.6667 1.0000 1.0000 0.6667 0.6667 0.6667]
cross_val_score: 0.7333
```
**👉 Say:** "Ten fold accuracies (each k/9, so they vary), mean 0.7333 = the training score. Then 80 % is used to fit the saved model and the held-out 20 % is saved for predict."

**❓ Q&A**
- *Which validation tool?* — `cross_val_score` over the **whole pipeline** with `ShuffleSplit(n_splits=10, test_size=0.2)`.
- *Where do 45 / 36 / 9 / 10 come from?* — 45 trials (3 runs × ~15); 20 % = 9; 80 % = 36; 10 = chosen rounds.
- *What is a "fold"?* — One train/test split + its score; 10 folds averaged.
- *Why 10 rounds, not 1?* — A single split is luck-dependent (folds 0.44–1.0); averaging stabilizes it.
- *Does it train once and test at the end?* — No: a **fresh CSP+LDA each round**, fit on 36, scored on 9, discarded.
- *Then what's the saved model?* — A separate `train_test_split` fit on 80 %, saved with the held-out 20 % — so `train` does two independent things (score A + deployable model B).
- *Why `ShuffleSplit`, not `KFold`?* — Repeated independent random 80/20 splits; both valid.
- *What does `fit` actually learn?* — CSP learns `filters_` (4×64); LDA learns direction `w` and bias `b`.
- *Is there leakage?* — No: CSP is the first pipeline step, refit per fold on training data only.
- *0.73 here but the gate is 60 %?* — 60 % is the **109-subject × 6-experiment mean (0.658)**, not one subject.
- *Reproducible?* — `TPV_SEED=42` fixes splits; unseeded they vary slightly.

→ **Yes**

---

## 2.4 Predict
**📋 Criterion (verbatim, evaluation sheet):**
> *"There is a predict mode, which also uses validation tools. The prediction output is displayed (the id of the output class is enough)."*

**📚 Concept — what you need to know:**
- The saved `.joblib` holds the **pipeline (fit on 80 %), the held-out 20 % (`X_test`, `y_test`), and meta**. `predict` loads it and replays those held-out trials — **never seen in training** — so the accuracy is honest.
- **Models are subject-specific**: each is fit per subject (EEG differs per brain/cap), so `predict 4 14` needs `train 4 14` first. A brand-new person just needs a short **calibration** (train a fresh model on a little of their labeled data).

**🗣️ Say:** "`predict` loads the saved model and streams the held-out 20 % (9 trials) one at a time, comparing prediction to truth — no retraining."
**🖥️ Show:** `python mybci.py 4 14 predict`
```
epoch 00:  [2]  [2] True
...
Accuracy: 0.6667
```
**👉 Say:** "Each line is **[prediction] [truth] match**; ids 1 = imagined hands (T1), 2 = imagined feet (T2). These 9 were never seen in training; 6/9 = 0.6667."

**❓ Q&A**
- *Where are the 'validation tools'?* — Be precise: "validation tools" = the *scoring method*, not the data. `predict` evaluates the model on the **held-out set** `train` saved (trials it never saw) and computes **accuracy vs ground truth** — that's the validation. It does *not* re-run `cross_val_score` (that's `train`'s job); here the validation is the held-out accuracy.
- *Why only 9 trials?* — The 20 % `train` set aside for this subject/experiment.
- *0.6667 vs train's 0.7333?* — Train = 10-fold average; this = one fixed 9-trial test. Different estimator.
- *Does predict retrain?* — No, only loads the `.joblib`; inference = CSP transform + one LDA dot product.
- *Same subject/run as train?* — Yes; the artifact is named per subject/run and predict errors if it's missing.
- *Can you predict a person not in the 109?* — Yes, with a short calibration (train a fresh model on their labeled data). The 109 isn't a fixed menu.
- *Why subject-specific, not one universal model?* — Brains/caps differ; per-subject (calibrated) models are the BCI standard and usually beat a generic one. Training is fast/analytic, not a deep net.

→ **Yes**

---

## 2.5 Realtime
**📋 Criterion (verbatim, evaluation sheet):**
> *"The prediction is made as the data is streamed to the processing pipeline. The program outputs the result between 0 and 2 seconds after the event was triggered."*

**📚 Concept — what you need to know:**
- We feed the pipeline **one event (2-second epoch) at a time**, like a stream. Inference is a matrix multiply + variance + a dot product → **sub-millisecond**. A `assert latency < 2.0` enforces the budget.
- The "2 seconds" is mostly **collecting the 2-second window** after the event (TMIN=0…TMAX=2.0); the compute is negligible.

**🗣️ Say:** "predict streams one 2-second epoch at a time; measured latency is ~0.1–0.5 **ms** — a ten-thousandth of the 2-second budget — and `assert latency < 2.0` enforces it. The data is pre-recorded, so this is a **simulated stream** (replaying held-out epochs one by one) rather than live hardware; I used a loop instead of `mne-realtime`."
**🖥️ Show:** `python mybci.py 4 14 predict` — every line ends with the inference latency, e.g. `epoch 00:  [2]  [1] False  0.42 ms`. Each event is ~0.1–0.4 ms, far under the 2-second budget.

**❓ Q&A**
- *Is it a real live stream?* — A per-event simulation over pre-recorded data; the processing model (one event, 2 s budget) is identical to live.
- *Why so fast?* — Inference has no iterative computation.
- *What dominates the 2 s?* — Collecting the 2-second window; compute is ~0.5 ms.
- *Where is 2 s guaranteed / how do you prove the timing?* — `predict` measures each inference with `time.perf_counter` and **prints it as a latency column** (~0.1–0.4 ms); `assert latency < LATENCY_BUDGET_S` (2.0 s) crashes if any event ever exceeds the budget; and the epoch itself is the 2-second post-event window.
- *Why not mne-realtime?* — A loop satisfies "streamed, within 2 s" with no extra real-time dependency.

→ **Yes**

---

## 2.6 Integration
**📋 Criterion (verbatim, evaluation sheet):**
> *"Implementation was integrated to sklearn pipeline, inheriting from the baseEstimator and transformerMixin classes of sklearn."*

**📚 Concept — what you need to know:**
- **What sklearn is.** scikit-learn is Python's standard machine-learning library. It gives you (a) ready-made **parts** (classifiers like LDA, data splitters, transformers) and (b) **tools** that drive them (`Pipeline`, `cross_val_score`, `GridSearchCV`). The key idea: every part follows the **same interface ("contract")**, so one tool can operate any part the same way. `Pipeline` = an assembly line that chains parts (CSP stage → LDA stage); each stage only has to honor that contract.
- **Inheritance = getting a parent's abilities for free.** `class MyCSP(TransformerMixin, BaseEstimator)` makes MyCSP a *child* of those two classes, so it automatically gains their methods without rewriting them. Those two parents *are* the sklearn contract:
  - **BaseEstimator** — the base class every sklearn estimator inherits. It provides **`get_params()` / `set_params()`**: a standard way to read and re-set a part's hyperparameters (here `n_components`, `reg`, `solver`). This is the "ID card" that lets sklearn **clone** it (make a fresh, unfitted copy), **tune** it (GridSearchCV sweeps params via `set_params`), and **inspect** it.
  - **TransformerMixin** — a small add-on class (a "**mixin**" = grants one capability). It gives the **transformer** role: once you write `fit` and `transform`, it hands you **`fit_transform`** for free and marks the class as something that *reshapes* data (input → output), e.g. CSP's 64×321 → 4. (A classifier would use `ClassifierMixin` + `predict` instead.)
- **Why it matters (not a formality).** Because MyCSP honors this contract, `cross_val_score` can **`clone` it and re-`fit` it on each fold's training data only** — which is exactly what prevents leakage. So the inheritance is the foundation of honest scoring, not bureaucracy.

**🗣️ Say:** "My CSP inherits `TransformerMixin` and `BaseEstimator`, so it has the `fit`/`transform` contract and drops straight into `Pipeline([('csp', MyCSP()), ('clf', LDA())])`. That inheritance is exactly what lets `cross_val_score` clone and refit it per fold — so it's the foundation of honest scoring."
**🖥️ Show:** `isinstance(csp, BaseEstimator)=True`, `isinstance(csp, TransformerMixin)=True`, `get_params()={...}`, has `fit_transform`, `clone()` works, plugs into the Pipeline.

**❓ Q&A**
- *What is inheritance?* — A child class automatically gets the parent's methods (here, the sklearn contracts) without rewriting them.
- *Why both base classes?* — BaseEstimator = parameter management (clone/tune/inspect); TransformerMixin = transformer contract. Both → a full sklearn component.
- *What is a "mixin"?* — A small helper class that grants one capability through inheritance; `TransformerMixin` adds `fit_transform` and the transformer role on top of your own `fit`/`transform`.
- *What is sklearn in one line?* — Python's standard ML library: ready-made parts + tools (Pipeline, cross_val_score), all sharing one interface so any tool can drive any part.
- *What does `clone` do, and why does CV need it?* — It copies an unfitted estimator from its params; `cross_val_score` clones the pipeline each fold and refits on training data only (no leakage).
- *Why does `__init__` do no computation?* — sklearn contract: store hyperparameters only; learning happens in `fit`, else `clone` breaks.
- *Transformer vs classifier?* — A transformer (`fit`/`transform`) reshapes data (CSP: 64×321→4); a classifier (`fit`/`predict`) outputs a label (LDA).
- *Did you avoid the library CSP?* — Yes; `mne.decoding.CSP` is only a parity reference in tests.

→ **Yes**

---

## 2.7 Implementation — dimensionality reduction (high-value)
**📋 Criterion (verbatim, evaluation sheet):**
> *"A dimensionality reduction algorithm is implemented, the subject talks about PCA and CSP but other algorithms performing a dimensionnality reduction are feasible. Check that the student has a general understanding of the algorithm. It is allowed to use functions from libs like numpy or scipy for some tasks : the eigenvalues decomposition, singular values decompositon and covariance matrix estimation."*

**📚 Concept — built up from scratch (the core of the project, `csp.py`):**

*Start with the problem:*
- **Why one trial has ~20,000 numbers.** A trial is **64 channels × 321 time samples = 20,544 numbers** (64 microphones, each read 321 times over the 2 seconds). That count is the "dimension".
- **Why we must shrink it = dimensionality reduction.** With only ~45 trials per subject, 20,544 numbers is *far* too many to learn from — a classifier would just **memorize** (overfit), not generalize. And we don't need them all: from ERD we know the answer is in **how big the oscillation is**, not in every raw value. **Dimensionality reduction** = squeezing those ~20k numbers into a few that keep the discriminative info — here, **4**.
- **What CSP is.** **CSP (Common Spatial Patterns)** is the reducer I implement: it finds the best **combinations of channels** that make the two classes differ most in oscillation size.

*How 20,544 → 4 (think of a trial as a table: **64 rows = channels × 321 columns = time = 20,544 cells**):*
- **Cut 1 — channels 64 → 4.** A **recipe** = 64 weights, one per channel. Blending all 64 channels with it (multiply-and-sum, `@`) makes **one new "virtual channel"** (a row of 321) — like a DJ mixing 64 tracks into one. **One recipe → one row; 4 recipes → 4 rows** (`64×321 → 4×321`). And each recipe is a **detector**: its virtual channel wiggles big for one class, small for the other — usually **2 detect left, 2 detect right**.
- **Cut 2 — time 321 → 1.** For each virtual channel, measure **how big it wiggled** = its **variance** → one number; then normalize and take **log**. So `4×321 → 4`. **Total: 20,544 → 4** (a recipe eats 64 channels → 1 row; a variance eats 321 time samples → 1 number).

*How CSP finds those 4 recipes:* it keeps the 4 blends where the two classes differ **most** in wiggle size — the "**4 best one-sided detectors**." (Technically these are the top **eigenvectors** of `eigh(C1, C1+C2)`, each scored by **λ** = how one-sided it is; keep the 4 with λ farthest from 0.5. numpy/scipy are allowed for the covariance + eigendecomposition — everything else is from scratch.)

**📚 What the 4 numbers feed — LDA (the classifier):** each trial's 4 numbers are one **point** (~45 trials → ~45 points), and they form **two clouds** (one per class). **LDA draws a single straight line (a "fence") between the two clouds**, then classifies a new point by **which side of the fence it lands on**. (Its math `w = Σ⁻¹(μ1−μ0)` — the center-to-center direction, corrected by the clouds' spread — is the from-scratch bonus, item 2.11-D.) LDA is *not* dimensionality reduction — this item is only about the reducer (CSP).

**🗣️ Say:** "I implemented **CSP from scratch** in `csp.py` — imports are only numpy, `scipy.linalg.eigh` (explicitly allowed), `sklearn.base`, and my own `generalized_eigh`; no library CSP. The algorithm, in four steps: reduce because the answer is oscillation size and 45 trials can't fill 20k-D; build per-class covariances; solve `eigh(C1, C1+C2)` for the filters whose eigenvalue is farthest from 0.5; then blend with `@` and take log-variance → 4 numbers."
**🖥️ Show:** `pytest tests/test_csp_parity.py -v` → **PASSED** (matches `mne.decoding.CSP` within 0.05 on identical splits, ≥ 0.60 on a single subject).
**🗣️ Code tour:** `_class_cov` = `E@E.T` → trace-normalize (a loud trial doesn't dominate) → average → symmetrize → **shrinkage** (`reg=0.01`, keeps `eigh` well-conditioned). `fit` = C1,C2 → `eigh` → sort by `|λ-0.5|` → `filters_` (top 4×64). `transform` = `filters_ @ E` → variance → log.

**❓ Q&A**
- *What is an eigenvalue here?* — The fraction of a filter's oscillation belonging to one class; far from 0.5 = discriminative.
- *What is the `@`?* — Matrix multiply = multiply-and-sum; blends 64 channels into one virtual channel.
- *What's covariance here?* — A 64×64 table of how channels co-oscillate within a class; CSP's raw material.
- *Why a generalized eigenproblem (two matrices)?* — We maximize the **ratio** `wᵀC1w/wᵀ(C1+C2)w`, putting (C1+C2) in the denominator.
- *Why CSP over PCA?* — PCA is unsupervised (largest variance, fooled by artifacts); CSP is supervised (largest class difference).
- *Why log-variance, not variance?* — De-skews toward Gaussian and linearizes multiplicative differences.
- *Why trace-normalize / shrinkage?* — So one loud trial doesn't dominate; shrinkage keeps the covariance invertible/`eigh` stable on few/colinear channels.
- *`filters_` vs `patterns_`?* — filters_ maps signal→feature (analysis); patterns_ maps feature→scalp (visualization).
- *Why exactly 2 classes?* — CSP is a 2-class method; our six experiments are all binary, so no OvR needed.
- *Prove it's correct, not plausible?* — Parity test vs MNE (<0.05); the bonus Jacobi solver matches scipy to ~1e-14.
- *Did you use a forbidden function?* — No; the sheet allows numpy/scipy for eigendecomposition/SVD/covariance, and I even reimplemented those.

→ **Yes**

---

## 2.8 Score (the 60 % gate)
**📋 Criterion (verbatim, evaluation sheet):**
> *"There has to be a script executing training over each subject and computing the mean of scores over each subjects, by type of experiment runs. The mean of the resulting six means (corresponding to the six types of experiment runs) has to be superior or equal to 60%."*
> *(Rating)* *"Over 60% add a point for every 1%."* — *"Rate it from 0 (failed) through 5 (excellent)"*

**📚 Concept — what you need to know:**
- **The six experiments** (each binary): 0 exec L/R hand · 1 imagined L/R hand · 2 exec hands/feet · 3 imagined hands/feet · 4 exec vs imagined (hands) · 5 exec vs imagined (feet).
- The gate = for each experiment, average the cross-validated accuracy over **109 subjects** (6 means), then **average those 6**. Must be **≥ 0.60**. Per-subject scoring is the leakage-free `cross_val_score` from item 2.3.

**🗣️ Say:** "`python mybci.py` (no args) cross-validates all 109 subjects × 6 experiments, averages per experiment (6 means), then averages those 6. The full run is **0.658 ≥ 0.60 → pass**; at 65.8 % the rating caps at **5/5**."
**🖥️ Show:** `python scripts/validate_60.py` (run the **full** version at the defense; `--fast 5` is a preview).

**❓ Q&A**
- *How is 60 % computed?* — Per-experiment mean over subjects (6 numbers), then the mean of those 6 — not one flat average.
- *Why does `--fast 5` show ~0.71 but full is 0.658?* — The first 5 subjects are easy; the full 109 includes harder ones. Official = full.
- *What if a subject fails?* — `run_all` skips **only** data/IO errors (narrow catch); genuine algorithm bugs propagate, never hidden. I can rerun that subject's `train` in isolation.
- *Is the gate seeded?* — Unseeded it fluctuates slightly around 0.65; the bar is on the grand mean.
- *Why the per-subject variance?* — BCI is famously subject-variable; that's why the bar is a 109-subject mean and each subject uses 10-fold averaging.
- *Why is exp0 (L vs R hand) the weakest experiment (~0.57)?* — Both are hands on adjacent, near-symmetric motor cortex, so the contralateral difference is small; hands-vs-feet (lateral vs midline) is much easier (~0.71–0.84).
- *Time?* — ~2.5 min cached; first run also downloads ~3.1 GB.

→ **Yes / 5/5**

---

## 2.9 Bonus · Datasets
**📋 Criterion (verbatim, evaluation sheet):**
> *"Are there other datasets processed by the program ? Is the scoring on those datasets correct ? Try to assert this taking into account the noise and the general quality of the dataset compared to the one given in the subject."*

**📚 Concept — easy version (what / why / how / result):**
- **What I added.** A **second, completely separate EEG dataset** — **BCI Competition IV-2a**, a well-known motor-imagery benchmark — pulled in with the **moabb** library.
- **Why it's a good test.** It's built totally differently from PhysioNet: **22 electrodes (not 64), 250 samples/sec (not 160), 288 trials**. If my pipeline handles a *different* dataset with **no code changes**, that proves it's general, not hardcoded to PhysioNet.
- **How I process it.** `external.py` uses moabb to download IV-2a and reshape it into the **same `(trials, channels, time)` format** as my PhysioNet data, filtered to the **same 7–30 Hz** band (left- vs right-hand imagery). Then it feeds the **same MyCSP + LDA pipeline** — unchanged.
- **Why unchanged works.** MyCSP never assumes 64 channels — its covariance is `n_channels × n_channels`, so with 22 channels it just becomes 22×22 and adapts automatically.
- **Result.** **0.855** on IV-2a subject 1 — *higher* than PhysioNet's 0.658, which is **correct**: IV-2a is a clean, curated competition dataset (less noise), and the criterion says to judge "taking into account quality". Cleaner data → higher score.
- **Where it lives.** Downloads into the project's `mne_data/`; it's a **separate run** and does **not** touch the mandatory 60 % gate (PhysioNet only). (Under the hood I point moabb's `MNE_DATASETS_BNCI_PATH` at the project for the call, then restore it.)

**🗣️ Say:** "I added BCI IV-2a, a different dataset — 22 channels, 250 Hz — and my pipeline runs unchanged at **0.855**. It's higher than 0.658 because IV-2a is a curated competition dataset with less noise, exactly the quality caveat the sheet mentions."
**🖥️ Show:** `python scripts/bonus_demo.py` → `[G] 2nd dataset BCI IV-2a (subj 1) : 0.8552` (shape 288×22×1001).

**❓ Q&A**
- *Why does 64-ch code run on 22-ch data?* — No channel-count assumption anywhere; covariance and the eigenproblem size to the input.
- *Is 0.855 > 0.658 suspicious?* — No: cleaner data → higher score; the sheet asks to weigh quality.
- *Same preprocessing?* — Yes, 7–30 Hz via the moabb paradigm; same left-vs-right-hand imagery.
- *Where is the data stored?* — In the project's `./mne_data/MNE-bnci-data/` (~83 MB), gitignored, so it travels with the folder.
- *Merged with the gate?* — No, a separate run; the 60 % gate uses PhysioNet only.

→ **Yes**

---

## 2.10 Bonus · Feature engineering
**📋 Criterion (verbatim, evaluation sheet):**
> *"Try to evaluate the relevance of the preprocessing stage and how are the data feeded to the algorithm. The use of fourier or wavelet transform, and anything that transform the data before the processing is a plus."*

**📚 Concept — easy version (what it wants / what I built / how / result):**
- **What the criterion rewards.** Points for doing something *smarter* in how you transform the data before the algorithm — Fourier/wavelet or any clever pre-transform. It asks: "is your feature prep more than a plain filter?"
- **What I built: FBCSP (Filter-Bank CSP).** It runs on the **same PhysioNet data** as the mandatory pipeline — **only the CSP step is swapped** (data and LDA unchanged; not a new dataset). Plain CSP treats the whole 7–30 Hz band as **one lump**. FBCSP instead **splits 7–30 Hz into 4 narrow sub-bands** (8-12, 12-16, 16-20, 20-30 Hz) and runs a **separate CSP on each sub-band**, then glues the features together.
- **Why that's smarter.** The motor signal (ERD) is stronger at different frequencies for different people — some in **mu (8-12)**, some in **beta (13-30)**. One wide band blurs them; splitting gives **each sub-band its own detectors**, and the classifier learns **which sub-bands matter**. That's the "learns which frequencies matter" bonus the sheet points to (a frequency-domain transform, Fourier/wavelet family).
- **How the data is fed.** Each sub-band → band-pass filter → its own CSP → log-variance → **concatenate** → `4 bands × 2 components = 8 numbers` into the classifier. The band-pass lives inside `fit`/`transform`, so cross-val refits per fold (no leakage).
- **Result.** **0.822** here (vs plain CSP 0.844 on this subject). Honestly slightly lower — 8 features can mildly overfit ~45 trials — but it's a **strictly more expressive** method that can win on other data; it demonstrates a genuine, more sophisticated feature-engineering technique.

**🗣️ Say:** "Two layers of frequency transformation: the mandatory FIR band-pass is already one; the bonus **FBCSP** splits the band into four sub-bands and runs a CSP per band, so the classifier can weight which frequencies matter. Honestly, here it's 0.822 vs 0.844 — eight features mildly overfit ~45 trials — but it's strictly more expressive and a sound feature-engineering demo."
**🖥️ Show:** `[F] Filter-Bank CSP (4 sub-bands) : 0.8222` (plain CSP 0.8444).

**❓ Q&A**
- *What does FBCSP add over CSP?* — Per-sub-band spatial patterns, so the classifier learns band importance.
- *Why isn't it always better?* — More features → mild overfitting on ~45 trials; a sophistication demo, not a guaranteed win.
- *How is data fed?* — Each sub-band → its CSP → log-variance → concatenated (8 numbers) into the classifier.

→ **Yes**

---

## 2.11 Bonus · Implementations
**📋 Criterion (verbatim, evaluation sheet):**
> *"How deep did the student dig into his implementation ? ( Did he implement his own eigenvalues decomposition, SVD, or covariance matrix estimation ? ) ( Did he implement a complex dimensionality reduction algorithm ? ) Is there some kind of hyperparameter tuning or learning ? Did he implement his own classifier ?"*

**📚 Concept — easy version.** The criterion rewards **going deep** — writing the hard math yourself instead of only calling libraries. I did **three**:

**A. My own eigendecomposition — `jacobi.py`.**
- *What it is.* The eigendecomposition (`eigh`) is the heart of CSP — it finds the recipes. Normally you just call scipy; I wrote **my own** in pure numpy.
- *How (easy).* The **Jacobi** method repeatedly "rotates" the matrix to turn its off-diagonal numbers into zeros, one pair at a time, until only the diagonal is left → the **diagonal = the eigenvalues**, the accumulated **rotations = the eigenvectors**. It uses only `+, ×, sqrt, sign` — no `eig/eigh/svd`, no scipy. (The generalized problem `C1 w = λ(C1+C2)w` is first turned into a plain one by "whitening", then Jacobi runs twice.)
- *Proof.* Matches scipy to **~1e-14** (essentially identical). Turn on with `MyCSP(solver="jacobi")`.

**C. Hyperparameter tuning — `evaluate.tune`.**
- *What it is.* The pipeline has a knob — `n_components` (how many recipes; default 4). Instead of guessing, I use **`GridSearchCV`** to automatically try **4/6/8** and keep the best.
- *How (leakage-free).* It's **nested** cross-validation: for each candidate, CSP is refit inside each inner fold, so no test data leaks. The official 60 % gate stays fixed at 4 (fair, comparable score); tuning is a demo.

**D. My own classifier — `own_lda.py`.**
- *What it is.* LDA is the classifier that draws the fence. Normally you call sklearn's LDA; I wrote **my own** in numpy.
- *How (easy).* Compute the two class **centers** μ0, μ1, the fence **direction** `w = Σ⁻¹(μ1−μ0)` (center-to-center, corrected by the clouds' spread), then classify a point by the **sign of `w·x + b`** — using only `np.linalg.solve`.
- *Proof.* **100 % identical predictions** to sklearn's LDA.

(+ **F. FBCSP** also counts here as a "complex dimensionality reduction".)

**🖥️ Show:** `python scripts/bonus_demo.py` prints all of these in one run:
```
[A] from-scratch Jacobi eigensolver : 0.8444   (standard eig vs numpy 5.33e-14, generalized vs scipy 1.55e-15)
[C] tuned {'csp__n_components': 8}   -> cv 0.8444
[D] from-scratch OwnLDA classifier  : 0.7889   (100% agreement with sklearn LDA)
```

**🗣️ Say:** "I dug to the bottom: my own eigensolver (Jacobi, matching scipy to ~1e-14), leakage-free hyperparameter tuning, and my own LDA classifier (100 % agreement with sklearn) — beyond what the sheet even requires."

**❓ Q&A**
- *Really no eigendecomposition library?* — `jacobi.py` uses only basic ops; matches scipy ~5e-14.
- *How does Jacobi work?* — Rotations zero off-diagonals until diagonal = eigenvalues; whitening reduces the generalized problem, then Jacobi twice.
- *Is the tuning leakage-free?* — Yes; CSP is refit inside each inner fold.
- *Why keep the gate at 4 components?* — Fair, comparable scoring; tuning is a demo.
- *Does OwnLDA match sklearn?* — 100 % identical predictions (same pooled covariance + Σ⁻¹(μ1−μ0)).

→ **Yes**

---

## 2.12 Ratings & Conclusion
**📋 Verbatim:** *"Don't forget to check the flag corresponding to the defense"* — *"Ok / Outstanding project"*; flags *"Empty work / Incomplete work / Cheat / Crash / Forbidden function"*; *"Leave a comment on this evaluation ( 2048 chars max )"*.

**🗣️ Say:** "No negative flag applies — in particular **Forbidden function**: the sheet allows numpy/scipy for eigendecomposition/SVD/covariance, and I reimplemented even those. Mandatory passes at 0.658 and all five bonuses are implemented and verified, so I'd argue **Outstanding project**."

**Closing line:** "PhysioNet's 64 channels are band-passed to 7–30 Hz to expose ERD, compressed by a from-scratch CSP into 4 numbers, classified by LDA, scored with leakage-free cross-validation to a 109-subject × 6-experiment mean of **0.658 (≥ 60 %)**; all five bonuses — including a from-scratch eigensolver — are implemented."

→ **Outstanding** ⭐

---

# 3. Command runbook (in order)

```bash
source .venv/bin/activate
python scripts/validate_60.py              # full 60% gate → 0.658  (run BEFORE defense too)
pytest -m "not network"                    # fast unit tests
pytest tests/test_csp_parity.py -v         # scratch CSP vs mne, within 0.05
python scripts/visualize.py 4 14           # raw vs 7–30Hz + PSD
python mybci.py 4 14 train                 # cross_val_score + save model
python mybci.py 4 14 predict               # stream held-out trials
python mybci.py                            # 6-experiment means → 0.658
python scripts/bonus_demo.py               # bonuses A,C,D,F,G in one shot
```

---

# 4. Final checklist

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

> **Each item above is self-contained — concept, criterion, demo, and Q&A together. Walk them top to bottom and you can defend the whole project.**
