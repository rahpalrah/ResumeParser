# RSNA Knee Abnormalities Detection — CARE-Net pipeline

A complete, Kaggle-GPU-only pipeline for the RSNA Knee Abnormalities Detection
competition (12 binary findings per study, macro AUC-ROC).

Everything here is written to run inside Kaggle notebooks. Nothing in this repo
runs the competition data locally — the dataset is 570 GB and lives only on
Kaggle.

---

## 0. What the approach is, in one paragraph

Three facts about *this* competition drive the design:

1. **Reports exist for every training study, labels for only a few.** So the
   first model trained is not an image model at all — it is a multilingual text
   model that reads the reports and manufactures soft targets for the ~90% of
   studies that have no gold labels. The test set has no reports, so this model
   never runs at inference; it exists purely to distil into the image model.
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
on the first submission. What a single fold of the config below realistically
produces is roughly **0.87–0.91 local CV**, and the gap to the top of the board
is closed by scale (more slices, larger backbones, more folds, 3–5 different
backbones ensembled) rather than by a different idea. Kaggle's 9-hour GPU cap
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
| `knee_text.py` | multilingual report lexicon (clause-scoped, negation-aware) |
| `selftest.py` | 15 checks on synthetic data — run after editing any module |
| `step0_setup.py` | environment + data sanity check |
| `step1_reports.py` | multilingual rule lexicon over the reports |
| `step2_text_teacher.py` | XLM-R report teacher → soft labels for every study |
| `step3_preprocess.py` | 570 GB DICOM → a few GB of JPEG sprite sheets |
| `step4_train.py` | two-stage training (distil, then gold fine-tune) |
| `step5_submit.py` | offline inference → `submission.csv` |

Each `stepN_*.py` is a notebook written as `# --- CELL n ---` blocks. Paste one
block per Kaggle cell.

Before uploading anything, run `python selftest.py` locally (CPU, ~30 s, needs
only numpy/pandas/opencv/torch/timm). It builds a fake sprite cache, runs the
dataset, the model, the loss, the EMA, the mirror TTA, two optimiser steps, a
checkpoint round trip, the metric, and the report lexicon. It catches shape and
masking bugs in seconds instead of six GPU hours into step 4:

```
[1] fake cache: 28 series / 8 studies
...
[13] two optimiser steps ran; head weight moved by 7.13e-04
[14] eval produced (8, 12) predictions, macro AUC 0.602
[15] checkpoint save/reload reproduces identical predictions
ALL SELF-TESTS PASSED
```

## 3. Packaging the code for Kaggle

Do this once, and re-do it whenever you edit a `knee_*.py`.

1. On your machine: `git clone https://github.com/rahpalrah/ResumeParser`
2. Create a Kaggle Dataset named **`knee-code`** containing just
   `knee_common.py`, `knee_model.py`, `knee_data.py`.
3. In any notebook with internet ON you can skip the dataset and use:
   ```python
   !git clone -q https://github.com/rahpalrah/ResumeParser /kaggle/working/repo
   !cp /kaggle/working/repo/rsna_knee/knee_*.py /kaggle/working/
   ```
   The submission notebook has internet OFF, so it must use the dataset.
4. Create a Kaggle Dataset **`knee-wheels`** for the offline notebook:
   ```bash
   pip download pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm -d wheels/
   ```
   Upload `wheels/`.

---

# The steps

## STEP 0 — sanity check (GPU, ~5 min, internet ON)

Run `step0_setup.py`. It prints the GPU, checks the DICOM decoders, summarises
the CSVs, and — the important part — times one full series read so you can size
step 3.

**Expected output (shapes will match, exact numbers will not):**

```
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
  Medial Meniscus   ~25%   ####################
  ...
  Fracture          ~2%    #

slices on disk: 30
volume (16, 256, 256)  dtype uint8  in 1.4s
sprite jpeg: 78.4 KiB -> projected cache = 2.1 GiB
projected preprocessing wall time on 4 procs: 2.6 h
```

**Gate:** if `volume` is `None`, a decoder is missing — install the wheels and
re-run cell 2 before going on. If the projected cache is over ~15 GiB, drop
`img_size` to 224 or `n_slices` to 12 in `Cfg`.

## STEP 1 — mine the reports (CPU, ~10 min, internet ON)

Run `step1_reports.py`. It builds a multilingual regex lexicon (anatomy terms ×
finding terms, in a proximity window, with negation cues) and scores it against
the gold labels.

The lexicon is clause-scoped and negation-aware, which is not cosmetic: reports
read "The ACL is intact. Tear of the lateral meniscus." — a plain keyword window
would call that an ACL tear. The 14 cases in `selftest.py` pin that behaviour
across English, Spanish, French, German, Italian and Portuguese phrasings.

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

## STEP 2 — the report teacher (GPU, ~1.5 h, internet ON)

Run `step2_text_teacher.py`. Five folds of XLM-R + the rule features, then
soft-label every unlabelled study.

**Expected output:**

```
labelled 4,xxx | unlabelled 3x,xxx
  fold 0 ep 0: macro AUC 0.93xxx
  fold 0 ep 2: macro AUC 0.96xxx
  ...
TEACHER out-of-fold (step 1 rules were 0.8xxx):
  macro AUC = 0.96 – 0.98
study_targets.parquet (Nstudies, 14)
mean soft prevalence vs gold prevalence
  ACL                soft 0.141   gold 0.139
  ...
```

**Gate:** teacher OOF ≥ 0.95, and the soft prevalences within ~±0.03 of the gold
prevalences. A teacher below ~0.93 will inject more noise than signal — raise
`EPOCHS` to 4 or `MAX_LEN` to 640 before continuing. If soft prevalence is far
*below* gold for a label, the teacher is under-calling it; that label will stay
weak in the image model too.

Save the notebook version and publish its output as a dataset named
**`knee-text-teacher`**.

## STEP 3 — build the sprite cache (CPU 12 h notebook, internet ON)

Run `step3_preprocess.py` **once per shard**, changing `SHARD` from 0 to
`NUM_SHARDS-1`. Each run decodes its slice of the series list and writes one
JPEG per series. Save a version after each run; each output becomes a dataset.

Only series that win one of the four slots are decoded, so roughly a third of
the corpus is skipped before any DICOM is touched.

**Expected output per shard:**

```
train: 27,xxx series in slots (of 4x,xxx total)
shard 0/8: 3,4xx series
     200/3400  ok=   200    2.1 min elapsed  ETA  33.4 min
    ...
failures: 0
cache size this shard: 0.26 GiB
laterality distribution: {1: 1800, 0: 1500, 2: 100}
round-trip: (16, 256, 256) uint8 min 0 max 255
```

**Gates:**
- `failures` under ~0.5% of the shard. A larger number usually means one
  transfer syntax is not decodable — re-check step 0's backend list.
- `laterality` class 2 (unknown) under ~30%. Higher is survivable (the model
  has an unknown token) but the compartment branch loses power.
- The displayed slices must look like knee MRI, not noise or all-black.

Set `SPLIT = "test"` and run one more time to cache the three example test
studies — useful for debugging step 5 quickly.

## STEP 4 — train CARE-Net (GPU, ~7 h per fold, internet ON)

Attach: competition data, `knee-code`, `knee-text-teacher`, and every step-3
shard output. Run `step4_train.py` once per fold (`FOLD = 0..4`).

Stage A distils from the report teacher over every study; stage B fine-tunes on
gold labels only at 0.3× LR. Validation is always gold-only.

**Expected output:**

```
cached series: 27,xxx over 3x,xxx studies
cache roots: ['/kaggle/input/knee-cache-s0/cache', ...]
gold 4,xxx | soft 3x,xxx
fold 0: train 3,4xx / valid 8xx
CARE-Net convnext_tiny.fb_in22k_ft_in1k: 30.7 M parameters

=== STAGE A: report distillation over all studies ===
    A ep0 200/4800 loss 0.3412 6.2 min
  [A] epoch 0: loss 0.2611 | val macro AUC 0.81xxx | 121.4 min
  [A] epoch 1: loss 0.2233 | val macro AUC 0.84xxx | 120.9 min

=== STAGE B: gold fine-tune ===
  [B] epoch 0: loss 0.2104 | val macro AUC 0.86xxx | 14.1 min
  ...
  [B] epoch 5: loss 0.1588 | val macro AUC 0.88xxx | 14.0 min

FOLD 0 final (with mirror TTA): 0.89xxx
    ACL                0.94xx
    Medial Meniscus    0.90xx
    Medial OA          0.85xx
    Fracture           0.78xx
```

**Gates:**
- Stage A epoch 0 must already beat ~0.75. If it sits at ~0.50 the cache is not
  resolving — check `cache roots` and that `cached series` is non-zero.
- Stage B must improve on stage A's best. If it does not, the teacher is too
  noisy: lower `pseudo_weight` to 0.3.
- Per-label: `Medial OA` and `Lateral OA` should not be near-identical. If they
  are, the compartment branch is not contributing — confirm laterality is mostly
  known and that coronal series are present.

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
   once and using a stronger English model, raises the ceiling of stage A.

## 5. Licensing note

Competition rules require winning solutions to be open-sourced under CC-BY-NC
4.0, and all external data/models to be freely and publicly available. Both
pretrained backbones used here (timm ImageNet weights, XLM-R) satisfy that.
