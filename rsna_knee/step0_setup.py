# =============================================================================
# STEP 0 - environment + data sanity check.        Kaggle notebook, GPU T4 x2.
# Internet: ON.  Runtime: ~5 minutes.  Run this BEFORE anything else.
# =============================================================================

# --- CELL 1 -----------------------------------------------------------------
# Locate the step-00 bootstrap output.  Attached notebook outputs land under an
# unpredictable folder name, so find it by content rather than by name.
import os, sys, glob
_c = glob.glob("/kaggle/input/*/knee_common.py") + glob.glob("/kaggle/working/knee_common.py")
assert _c, "Attach the step-00 notebook output (Add Data -> Your Work -> Notebook Output)"
CODE_DIR = os.path.dirname(_c[0])
sys.path.insert(0, CODE_DIR)
print("code from:", CODE_DIR)

import time
import numpy as np, pandas as pd, torch
import knee_common as kc

COMP = "/kaggle/input/rsna-knee-abnormalities-detection"

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.device_count(), "device(s)")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory/2**30:.1f} GiB")

print("\nDICOM decode backends:")
st = kc.check_dicom_backends()
missing = [k for k in ("libjpeg", "openjpeg", "gdcm") if not st.get(k)]
if missing:
    print("\n  -> install the missing decoders (internet ON):")
    print("     !pip install -q pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm")

# --- CELL 3 -----------------------------------------------------------------
train = pd.read_csv(f"{COMP}/train.csv")
train_series = pd.read_csv(f"{COMP}/train_series.csv")
test = pd.read_csv(f"{COMP}/test.csv")
test_series = pd.read_csv(f"{COMP}/test_series.csv")
sub = pd.read_csv(f"{COMP}/sample_submission.csv")

print("train        ", train.shape)
print("train_series ", train_series.shape)
print("test         ", test.shape, "| test_series", test_series.shape)
print("submission   ", list(sub.columns))

labelled = train[kc.LABELS].notna().all(axis=1)
print(f"\nstudies with gold labels : {int(labelled.sum()):,} / {len(train):,} "
      f"({100*labelled.mean():.1f}%)")
print(f"studies with a report    : {int(train['Report'].notna().sum()):,}")

print("\nprevalence among labelled studies")
prev = train.loc[labelled, kc.LABELS].mean().sort_values(ascending=False)
for k, v in prev.items():
    print(f"  {k:<18s} {v:6.3%}  {'#' * int(v * 120)}")

print("\nseries descriptors")
print(train_series.groupby(["Anatomical_Plane", "Fluid_Sensitive",
                            "Fat_Suppression"]).size().to_string())
print("\nseries per study:",
      train_series.groupby("StudyInstanceUID").size().describe().to_string())

# --- CELL 4 -----------------------------------------------------------------
# Time a single-series read end to end.  This number decides how you shard the
# preprocessing step: total_series * seconds_per_series / 4 workers < 11 hours.
cfg = kc.Cfg(comp_dir=COMP)
one = train_series.iloc[0]
sdir = f"{COMP}/train_series/{one.StudyInstanceUID}/{one.SeriesInstanceUID}"
print("slices on disk:", len(glob.glob(sdir + "/*.dcm")))

t0 = time.time()
vol = kc.load_series_volume(sdir, cfg.n_slices, cfg.img_size)
dt = time.time() - t0
print(f"volume {None if vol is None else vol.shape}  dtype "
      f"{None if vol is None else vol.dtype}  in {dt:.2f}s")

nbytes = kc.write_sprite("/kaggle/working/_probe.jpg", vol, cfg.grid_w, cfg.jpeg_quality)
print(f"sprite jpeg: {nbytes/1024:.1f} KiB "
      f"-> projected cache = {nbytes*len(train_series)/2**30:.2f} GiB")
print(f"projected preprocessing wall time on 4 procs: "
      f"{dt*len(train_series)/4/3600:.1f} h")

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 3))
for i in range(min(6, cfg.n_slices)):
    plt.subplot(1, 6, i + 1); plt.imshow(vol[i * (cfg.n_slices // 6)], cmap="gray")
    plt.axis("off")
plt.suptitle("STEP 0 sanity: one preprocessed series")
plt.show()
