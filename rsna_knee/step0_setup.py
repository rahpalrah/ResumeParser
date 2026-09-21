# =============================================================================
# STEP 0 - environment + data sanity check.        Kaggle notebook, GPU T4 x2.
# Internet: ON.  Runtime: ~5 minutes.  Run this BEFORE anything else.
# =============================================================================

# --- CELL 1 -----------------------------------------------------------------
# Bootstrap.  Uses the step-00 output when it is attached, and fetches the
# modules itself when it is not - so this notebook runs standalone with internet
# ON, and off the attached output when internet is OFF.
import os, sys, glob, subprocess

REPO = "https://github.com/rahpalrah/ResumeParser"
BRANCH = "claude/knee-mri-abnormalities-kaggle-6jzvyk"

def _locate_code():
    hits = (glob.glob("/kaggle/input/*/knee_common.py")
            + glob.glob("/kaggle/working/knee_common.py"))
    return os.path.dirname(hits[0]) if hits else None

CODE_DIR = _locate_code()
if CODE_DIR is None:
    print("step-00 output not attached - cloning the modules (needs internet ON)")
    subprocess.run(["rm", "-rf", "/kaggle/tmp/repo"], check=False)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO,
                    "/kaggle/tmp/repo"], check=True)
    subprocess.run("cp /kaggle/tmp/repo/rsna_knee/*.py /kaggle/working/",
                   shell=True, check=True)
    CODE_DIR = _locate_code()
assert CODE_DIR, ("Could not obtain the modules. Either turn internet ON, or run "
                  "step 00 and attach its output (Add Data -> Your Work -> "
                  "Notebook Output).")
sys.path.insert(0, CODE_DIR)
print("code from:", CODE_DIR)

import time
import numpy as np, pandas as pd, torch
import knee_common as kc

COMP = kc.find_comp_dir()          # autodetected: never hard-code the slug
print("competition data:", COMP)

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
# Time the real read path on a random sample of series.  One series is far too
# noisy to plan from: slice counts run 20-300 and transfer syntaxes decode at
# very different speeds.  This number decides how you shard step 3:
#     n_series * seconds_per_series / 4 workers  <  11 hours per shard
cfg = kc.Cfg(comp_dir=COMP)
TRAIN_ROOT = kc.find_series_root(COMP, "train")
N_PROBE = 5

sample = train_series.sample(N_PROBE, random_state=0)
times, sizes, first_vol = [], [], None
for _, row in sample.iterrows():
    sdir = os.path.join(TRAIN_ROOT, row.StudyInstanceUID, row.SeriesInstanceUID)
    n_dcm = len(glob.glob(sdir + "/*.dcm"))
    t0 = time.time()
    vol = kc.load_series_volume(sdir, cfg.n_slices, cfg.img_size)
    dt = time.time() - t0
    if vol is None:
        print(f"  DECODE FAILED  {n_dcm} slices  {row.Anatomical_Plane}")
        print(f"    {sdir}")
        continue
    kb = kc.write_sprite("/kaggle/working/_probe.jpg", vol, cfg.grid_w,
                         cfg.jpeg_quality) / 1024
    times.append(dt); sizes.append(kb)
    first_vol = first_vol if first_vol is not None else vol
    print(f"  {n_dcm:>4} slices -> {vol.shape}  {dt:5.2f}s  {kb:6.1f} KiB  "
          f"{row.Anatomical_Plane}")

assert times, (
    "Every probe series failed to decode. Install the DICOM decoders "
    "(cell 2 prints the pip line) and re-run this cell.")

t_med = float(np.median(times))
kb_med = float(np.median(sizes))
n_series = len(train_series)
print(f"\nmedian {t_med:.2f}s and {kb_med:.1f} KiB per series over "
      f"{len(times)}/{N_PROBE} successful probes")
print(f"projected cache for all {n_series:,} series : "
      f"{kb_med * n_series / 2**20:.2f} GiB")
print(f"projected step-3 wall time on 4 procs      : "
      f"{t_med * n_series / 4 / 3600:.1f} h total")
print(f"  -> with NUM_SHARDS = 8 that is "
      f"{t_med * n_series / 4 / 3600 / 8:.1f} h per shard "
      f"(must stay under ~11 h)")

import matplotlib.pyplot as plt
step = max(1, cfg.n_slices // 6)
plt.figure(figsize=(12, 3))
for i in range(6):
    plt.subplot(1, 6, i + 1)
    plt.imshow(first_vol[min(i * step, cfg.n_slices - 1)], cmap="gray")
    plt.axis("off")
plt.suptitle("STEP 0 sanity: these must look like knee MRI, not noise")
plt.show()
