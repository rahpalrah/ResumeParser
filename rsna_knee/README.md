# RSNA Knee Abnormalities Detection — CARE-Net pipeline

A complete, Kaggle-GPU-only pipeline for the RSNA Knee Abnormalities Detection
competition (12 binary findings per study, macro AUC-ROC).

**Everything runs on Kaggle.** You need no machine of your own — not even to
get this code onto Kaggle. Step 00 clones this repo from inside a Kaggle
notebook and hands every later step the modules and the offline pip wheels.

---

## 0. What the approach is, in one paragraph

Three facts about *this* competition drive the design:

1. **Reports exist for all 4,407 training studies; gold labels exist for ~58.**
   Measured, not assumed — step 0 prints it. That is 1.3%, which rules out
   supervised fine-tuning and rules out cross-validation as a selection signal.
   So the first model trained is not an image model at all: the step-1 rule
   lexicon labels every report, a multilingual text model is fitted to *those*
   labels (self-training on thousands of noisy examples, not 58 clean ones),
   and its predictions become the image model's targets. The test set has no
   reports, so this model never runs at inference.
2. **`test_series.csv` gives plane / fluid-sensitivity / fat-suppression at test
   time.** So the study-level head is *told* what each series is instead of
   guessing. Each series token carries learned embeddings for its role.
3. **Five of the twelve labels are compartment-specific** (Medial vs Lateral
   meniscus, Medial vs Lateral OA, MCL). In coronal and axial slices those
   compartments are the left and right halves of the image, modulo knee
   laterality. A dedicated branch pools the halves separately and routes them to
   those labels, with a laterality token saying which half is medial — and a
   mirror augmentation that flips the token along with the pixels.

That is **CARE-Net**: Compartment-Aware, Report-distilled Ensemble Network.

```
DICOM series ──▶ 2.5D slices ──▶ CNN backbone ──┬─▶ slice transformer ─▶ series token
                                                │        + slot/plane/fluid/fat embeddings
                                                │                    │
                                                │          study transformer (CLS + laterality)
                                                │                    │
                                                │            coupled 12-label head ──▶ logits
                                                └─▶ medial / lateral half-pooling ──────┘
                                                              (compartment branch)
```

## 1. Honest expectations

The public top is **0.956 macro AUC**. Do not expect this pipeline to land there
on the first submission.

And be clear about what any local number means here: with ~58 gold studies, a
"local CV" figure has error bars of roughly ±0.05 and several labels will be
undefined outright (MCL has ~9 positives in total). **The leaderboard is your
validation set.** The gold studies catch gross breakage — a model at 0.5, a
flipped label — and nothing finer. Every number this pipeline prints against
the teacher's targets measures agreement with the teacher, not with truth.

The gap to the top of the board is closed by scale (more slices, larger
backbones, more folds, 3–5 different backbones ensembled) rather than by a
different idea. Kaggle's 9-hour GPU cap
is the binding constraint, which is exactly why step 3 caches the pixels once
and every later run is cheap.

The parts of this design that are genuinely differentiated — report distillation
at full training-set scale, series-role conditioning, and the compartment branch
— are aimed at the labels where the leaderboard is actually won or lost
(Medial vs Lateral OA, and the rare Fracture / Synovitis columns). Treat the
numbers printed at each step as the gate: if a step's output does not look like
the "expected output" below, stop and fix it before spending GPU hours.

## 2. Files

| file | what it is |
|---|---|
| `knee_common.py` | constants, DICOM reader, sprite cache, slot assignment, metric |
| `knee_model.py` | CARE-Net, asymmetric soft-target loss, weight EMA |
| `knee_data.py` | study-level dataset, augmentation, multi-label folds |
| `knee_text.py` | report lexicon: 8 languages, clause-scoped, negation on both sides |
| `run_step.py` | one-cell runner: clones and execs a step, no paste to truncate |
| `selftest.py` | checks on synthetic data — run by step 00 |
| `step00_bootstrap.py` | clones the repo on Kaggle, self-tests, downloads wheels |
| `step0_setup.py` | environment + data sanity check |
| `step1_reports.py` | multilingual rule lexicon over the reports |
| `step2_text_teacher.py` | XLM-R report teacher → soft labels for every study |
| `step3_preprocess.py` | 570 GB DICOM → a few GB of JPEG sprite sheets |
| `step4_train.py` | two-stage training (distil, then gold fine-tune) |
| `step5_submit.py` | offline inference → `submission.csv` |

Each `stepN_*.py` is a notebook written as `# --- CELL n ---` blocks. Paste one
block per Kaggle cell.

**Or paste `run_step.py` instead — one short cell that clones this repo and
runs the step from the file.** These steps are long, and a truncated paste
fails with `SyntaxError: incomplete input` at whatever line the copy stopped,
which reads like a bug in the code and is not. The runner cannot be truncated
into something that silently half-works, and what executes is always current.
Settings are read from `globals()` first, so an override goes above the exec:

```python
SHARD = 1          # before the exec; the step will not overwrite it
STEP = "step3_preprocess.py"
```

Step 00 runs `selftest.py` for you on Kaggle (CPU, ~30 s). It builds a fake
sprite cache, runs the dataset, the model, the loss, the EMA, the mirror TTA,
two optimiser steps, a checkpoint round trip, the metric, and the report
lexicon. It catches shape and masking bugs in seconds instead of six GPU hours
into step 4:

```
[1] fake cache: 28 series / 8 studies
...
[13] two optimiser steps ran; head weight moved by 7.13e-04
[14] eval produced (8, 12) predictions, macro AUC 0.602
[15] checkpoint save/reload reproduces identical predictions
ALL SELF-TESTS PASSED
```

## 2b. Run order at a glance

Eight notebooks, all on Kaggle. Nothing runs anywhere else.

| # | notebook | accelerator | internet | runs | gate before moving on |
|---|---|---|---|---|---|
| 00 | `step00_bootstrap.py` | CPU | ON | 1× | `ALL SELF-TESTS PASSED`, 5 decoders `OK` |
| 0 | `step0_setup.py` | GPU | ON | 1× | `volume (16, 256, 256) uint8` — not `None` |
| 1 | `step1_reports.py` | CPU | ON | 1× | rule macro AUC ≥ 0.75, no label P < 0.6 |
| 2 | `step2_text_teacher.py` | GPU | ON | 1× | teacher gold AUC **beats** rules-alone |
| 3 | `step3_preprocess.py` | CPU | ON | 1× | `failures: 0`, unknown laterality < 30% |
| 4 | `step4_train.py` | GPU | ON | 5× (`FOLD`) | select AUC > 0.75; gold AUC rising with it |
| 5 | `step5_submit.py` | GPU | **OFF** | per submission | `wrote submission.csv (1300, 13)` |

Only two variables are ever edited by hand: `SHARD` in step 3 and `FOLD` in
step 4.

## 3. How the notebooks find each other

Every notebook locates its inputs by **globbing for a filename**, never by a
dataset name:

```python
glob.glob("/kaggle/input/*/knee_common.py")        # the code, from step 00
glob.glob("/kaggle/input/*/study_targets.parquet") # the targets, from step 2
glob.glob("/kaggle/input/*/cache")                 # the sprites, from step 3
glob.glob("/kaggle/input/*/carenet_f*.pt")         # the weights, from step 4
```

So you never have to name a dataset exactly right. Attach a previous
notebook's output with **Add Data → Your Work → Notebook Output** and it is
found. If a required input is missing the notebook raises an `assert` naming
the step to run, rather than failing halfway through.

Two things are handled for you and should never be typed by hand:

- **The code.** If the step-00 output is not attached, each step clones the
  repo itself (internet ON). Only step 5 truly needs the attachment, because
  its internet is off.
- **The competition path.** `kc.find_comp_dir()` searches `/kaggle/input` for a
  file only this competition has, so neither the mount layout
  (`/kaggle/input/<slug>/` vs `/kaggle/input/competitions/<slug>/`) nor the
  slug spelling (`abnormality` vs `abnormalities`) can break a notebook. Every
  step prints the directory it resolved as its second line.

Each step ends with **Save Version → Save & Run All (Commit)**, which turns its
output into something the next step can attach.

Editing the code later: change the file in GitHub, re-run step 00, and re-attach
its newer output. Nothing else changes.

---

# The steps

## STEP 00 — bootstrap (CPU, ~3 min, internet ON)

New notebook, internet ON, no data attached. Run `step00_bootstrap.py`. It
clones this repo, copies the modules into `/kaggle/working`, runs the self-test,
and downloads the DICOM-decoder wheels that step 5 will need with no network.

**Expected output:**

```
copied knee_common.py
copied knee_data.py
copied knee_model.py
copied knee_text.py
...
[13] two optimiser steps ran; head weight moved by 7.13e-04
[14] eval produced (8, 12) predictions, macro AUC 0.602
[15] checkpoint save/reload reproduces identical predictions
ALL SELF-TESTS PASSED

17 wheels, 41.3 MiB
offline decoder check:
  OK   pydicom
  OK   pylibjpeg
  OK   libjpeg
  OK   openjpeg
  OK   gdcm
BOOTSTRAP COMPLETE.
```

**Gate:** `ALL SELF-TESTS PASSED`, and all five decoders `OK`. If the self-test
fails, stop — every later step inherits the same modules. Then Save Version.

## STEP 0 — sanity check (GPU, ~5 min, internet ON)

Attach the competition data and the step-00 output. Run `step0_setup.py`. It
prints the GPU, checks the DICOM decoders, summarises
the CSVs, and — the important part — times one full series read so you can size
step 3.

**Expected output (shapes will match, exact numbers will not):**

```
code from: /kaggle/working
competition data: /kaggle/input/competitions/rsna-knee-abnormality-detection
torch 2.x.x | cuda True | 2 device(s)
  gpu0: Tesla T4  14.7 GiB
  gpu1: Tesla T4  14.7 GiB

DICOM decode backends:
  OK   pydicom
  MISS pylibjpeg          <- install these before step 3
  ...
train         (N, 14)
train_series  (M, 5)
submission    ['StudyInstanceUID', 'ACL', 'MCL', ..., 'Fracture']

studies with gold labels : ~a few thousand / all studies
studies with a report    : all studies

prevalence among labelled studies
  Effusion          ~30%   ########################
  ...
  Fracture          ~2%    #

    30 slices -> (16, 256, 256)   1.41s    78.4 KiB  Sagittal
    28 slices -> (16, 256, 256)   1.22s    74.1 KiB  Coronal
  ...
median 1.30s and 76.2 KiB per series over 5/5 successful probes
projected cache for all 27,431 series : 1.99 GiB
projected step-3 wall time on 4 procs : 2.5 h total
  -> with NUM_SHARDS = 8 that is 0.3 h per shard (must stay under ~11 h)
```

**Gate:** every probe decodes (`5/5 successful`) and the displayed slices look
like knee MRI. `DECODE FAILED` on all five means a decoder is missing — install
the wheels from cell 1 and re-run. If the projected cache is over ~15 GiB, drop
`img_size` to 224 or `n_slices` to 12 in `Cfg`; if per-shard time exceeds ~11 h,
raise `NUM_SHARDS` in step 3.

## STEP 1 — mine the reports (CPU, ~10 min, internet ON)

Run `step1_reports.py`. It builds a multilingual regex lexicon (anatomy terms ×
finding terms, in a proximity window, with negation cues) and scores it against
the gold labels.

The corpus is only ~38% English. Measured by stopword probe: Romance ~21%,
Turkish ~12%, German ~9%, Russian ~5%. The lexicon covers all of them, and
three properties of it are load-bearing rather than cosmetic:

- **Clause scoping.** "The ACL is intact. Tear of the lateral meniscus." — a
  plain keyword window calls that an ACL tear.
- **Negation on both sides.** Turkish is verb-final: "Efüzyon izlenmedi" is
  "effusion was not observed". Checking only backwards takes every such report
  as positive for everything it mentions.
- **Nearest-structure attribution.** "Tear of the medial meniscus, intact
  lateral meniscus" is one positive, not two — a comma is no clause break, so
  both menisci see the same "tear" and only the nearer may claim it.

Two normalisation traps are handled in `norm_text`, and both fail silently
with zero matches rather than an error: Turkish dotless ı has no NFKD
decomposition, and NFKD *does* decompose Cyrillic (й→и, ё→е), so "Бейкера"
normalises to "беикера". The 26 cases in `selftest.py` pin all of this, and a
guard there asserts no pattern holds a character `norm_text` would alter.

**Expected output:**

```
report length (chars): mean ~800
language mix (crude stopword probe):
  'the '   ~55%
  ' de '   ~20%
  ...
anatomy-term coverage (share of reports mentioning the structure at all):
  ACL                ~78%
  ...
RULE-ONLY baseline on N labelled studies (this is the floor):
  macro AUC = 0.78 – 0.86
    ACL                0.88 ######################
    MCL                0.82 ###################
    ...
    Synovitis          0.65 #########
saved -> rule_features.parquet
```

**Gate:** macro AUC ≥ ~0.75, and no label with precision below ~0.6. Cell 3
prints the worst-precision label together with three of its false positives —
that is your edit list for `knee_text.py`. Rules do not need to be good, they
need to be *precise*; they become input features for step 2, and a noisy feature
is worse than a sparse one. Re-run `python selftest.py` after any lexicon
edit.

## STEP 2 — the report teacher (GPU, ~1 h, internet ON)

Run `step2_text_teacher.py`. It fits XLM-R to the **rule labels** over the
~4,349 studies with no gold annotation, selects on a held-out 10% of those, and
measures against the gold studies, which are held out of every fit.

This is self-training, not supervised learning. The rules are precise and
low-recall; a model fitted to thousands of their outputs generalises to the
paraphrases and languages the regexes miss, because those co-occur in the same
reports with the phrasings that do fire.

**Measured output** (this is what the run actually produced, not an estimate):

```
studies with at least one gold label : 58
fitting on 4,349 rule-labelled studies; 58 gold studies held out entirely
early-stopping split: 3,915 train / 434 val

rules alone, measured on the gold studies: 0.76659

  epoch 0: held-out rule AUC 0.83334 | GOLD AUC 0.68090 | 2.6 min
  epoch 1: held-out rule AUC 0.90620 | GOLD AUC 0.77370 | 2.6 min
  epoch 2: held-out rule AUC 0.93551 | GOLD AUC 0.80676 | 2.6 min
  epoch 3: held-out rule AUC 0.95043 | GOLD AUC 0.82388 | 2.6 min
  epoch 4: held-out rule AUC 0.95582 | GOLD AUC 0.82106 | 2.6 min
  epoch 5: held-out rule AUC 0.95653 | GOLD AUC 0.82104 | 2.6 min

gold-study macro AUC of each candidate target:
  rules alone    0.76659
  teacher alone  0.82104
  50/50 blend    0.82782
```

Per label, rules to teacher: Lateral OA +0.135, Fracture +0.116, Medial OA
+0.116, Synovitis +0.098, Contusion +0.072, Medial Meniscus +0.068. ACL
(-0.013) and Effusion (-0.006) are the only regressions, both inside noise at
this sample size.

Synovitis is the result worth noting. Step 1 named it in only 11.9% of reports
while 47% of gold studies are positive, and the conclusion there was that no
lexicon could recover a finding the report does not mention. The teacher lifted
it from 0.657 to 0.755 by reading the language that co-occurs with it. That is
self-training doing exactly the job it was added for.

**Gates:**
- Held-out rule AUC >= 0.93. Below that the model is not reproducing the
  lexicon on unseen reports, which is a tokenisation or truncation problem.
- **Teacher gold AUC > rules-alone gold AUC**, both printed. If self-training
  does not beat the lexicon it was trained from, it is adding noise - and it
  did exactly that when the rule scores were fed in as features, because a
  linear head then solves the task by copying them.
- Do not tune on the gold number. The checkpoint saved is epoch 5, selected on
  held-out rule AUC, even though gold peaked 0.003 higher at epoch 3 - that
  difference is one study changing rank.

Publish the output; step 4 finds `study_targets.parquet` by glob.

## STEP 3 — build the sprite cache (CPU 12 h notebook, internet ON)

Run `step3_preprocess.py` **once**. Step 0 measured the whole job at ~1 h on
4 procs, and this step decodes only the ~17,600 slot-assigned series, so a
single shard finishes inside the 12 h limit with hours to spare. Each run decodes its slice of the series list and writes one
JPEG per series. Save a version after each run; each output becomes a dataset.

Only series that win one of the four slots are decoded, so roughly a third of
the corpus is skipped before any DICOM is touched.

**Expected output per shard:**

```
train: ~17,600 series in slots (of 24,371 total)
shard 0/2: ~8,800 series
     200/8800  ok=   200    0.3 min elapsed  ETA  13.0 min
    ...
failures: 0
cache size this shard: ~2.0 GiB
laterality distribution: {1: 4400, 0: 4100, 2: 300}
round-trip: (16, 256, 256) uint8 min 0 max 255
```

**Gates:**
- `failures` under ~0.5% of the shard. A larger number usually means one
  transfer syntax is not decodable — re-check step 0's backend list.
- `laterality` class 2 (unknown) under ~30%. Higher is survivable (the model
  has an unknown token) but the compartment branch loses power.
- The displayed slices must look like knee MRI, not noise or all-black.

There is no `SPLIT = "test"` pass. Step 5 decodes DICOM directly and never
reads a sprite cache, so caching the test split would be dead work.

## STEP 4 — train CARE-Net (GPU, ~7 h per fold, internet ON)

Attach: competition data, `knee-code`, `knee-text-teacher`, and every step-3
shard output. Run `step4_train.py` once per fold (`FOLD = 0..4`).

It trains on the report teacher's targets over every study with no gold label,
selects checkpoints on a held-out fold of those same targets, and reports
against the gold studies with a per-cell mask. The gold studies never enter a
fit.

**Expected output:**

```
cached series: ~17,600 over 4,407 studies
cache roots: ['/kaggle/input/knee-cache-s0/cache', ...]
gold holdout 58 studies (696 annotated cells) | teacher-labelled 4,349
fold 0: train 3,479 / select-on 870 / gold 58
CARE-Net convnext_tiny.fb_in22k_ft_in1k: 30.7 M parameters

=== training on the report teacher's targets ===
    distil ep0 200/2175 loss 0.3412 6.2 min
  [distil] epoch 0: loss 0.2611 | select AUC 0.81xxx | gold AUC 0.72xxx | 31.4 min
  [distil] epoch 1: loss 0.2233 | select AUC 0.84xxx | gold AUC 0.75xxx | 31.0 min
  ...
  [distil] epoch 7: loss 0.1588 | select AUC 0.88xxx | gold AUC 0.79xxx | 31.2 min

FOLD 0 against the teacher targets, mirror TTA: 0.88xxx
  (how well it reproduces the teacher - the selection signal)

FOLD 0 against the 58 GOLD studies, mirror TTA:
  macro AUC = 0.79xxx
    ACL                0.85xx
    MCL                nan          <- too few positives at n=58
    Medial OA          0.74xx
```

**Gates:**
- Epoch 0 select-AUC above ~0.75. If it sits near 0.50 the cache is not
  resolving — check `cache roots` and that `cached series` is non-zero.
- Gold AUC should rise with select-AUC. If select-AUC climbs while gold AUC
  falls, the model is learning the teacher's mistakes rather than the anatomy;
  stop and improve the teacher.
- Per-label: `Medial OA` and `Lateral OA` should not be near-identical. If they
  are, the compartment branch is not contributing — confirm laterality is mostly
  known and that coronal series are present.
- Expect `nan` on some gold labels. With 58 studies a rare finding can have too
  few positives to define an AUC; that is honest reporting, not a bug.

Publish the fold checkpoints as a dataset **`knee-carenet`**.

For the **efficiency track**, set `EFFICIENCY = True` — a 192 px / 10-slice /
EfficientNetV2-T config that costs roughly a quarter of the inference time for
about 0.01–0.02 macro AUC.

## STEP 5 — submit (GPU, ~1.5 h, internet OFF)

Attach: competition data, `knee-code`, `knee-carenet`, `knee-wheels`. Run
`step5_submit.py`. Series are decoded straight from DICOM inside the DataLoader
workers; folds are rank-averaged, and the loop stops adding checkpoints when the
remaining time would not fit another pass.

**Expected output:**

```
1300 studies | 6,5xx series | 5 checkpoints
config: convnext_tiny.fb_in22k_ft_in1k 256 16
  ckpt 0 batch 0/325 0.1 min
  ...
                       StudyInstanceUID   ACL   MCL  ...
0  1.2.826.0.1.3680043.8.498.1004703...  0.41  0.22  ...
wrote submission.csv  (1300, 13)  in 84.3 min
```

**Gates:** the asserts in cell 4 must pass (column order identical to
`sample_submission.csv`, one row per test study, no NaN). Commit the notebook
and submit. A first submission that scores near 0.5 with a healthy local CV
almost always means the column order or the study order drifted.

---

## 4. Where to spend the next GPU hours, in order

1. **More folds and a second backbone.** `convnext_small` and
   `efficientnetv2_s` ensembled with `convnext_tiny` is the single biggest
   reliable gain. Rank-average, do not probability-average.
2. **More slices.** `n_slices` 16 → 24 helps the meniscus labels most; it costs
   a proportional amount of preprocessing and training time.
3. **A fifth and sixth slot.** Coronal T1 and axial non-fluid sequences, once
   the four-slot version is working.
4. **Per-label threshold-free calibration is pointless** — the metric is AUC, so
   only ranking matters. Spend the time on the rare labels instead: Fracture and
   Synovitis have the most headroom and the fewest positives.
5. **Better teacher.** A larger multilingual encoder, or translating the reports
   once and using a stronger English model, raises the ceiling of everything
   downstream. With 58 gold labels the teacher IS the supervision — it is worth
   more effort than the image model.

## 5. Licensing note

Competition rules require winning solutions to be open-sourced under CC-BY-NC
4.0, and all external data/models to be freely and publicly available. Both
pretrained backbones used here (timm ImageNet weights, XLM-R) satisfy that.
