# =============================================================================
# STEP 3 - build the sprite cache.                 Kaggle notebook, CPU (12 h).
# Internet: ON (only to install DICOM decoders).   Runtime: see step 0 estimate.
#
# 570 GB of DICOM cannot be read once per epoch.  This step reads every series
# exactly once and writes it as ONE grayscale JPEG "sprite sheet" holding the
# n_slices sampled slices in a grid.  The whole training set collapses to a few
# GB that fits in a Kaggle Dataset and loads in ~2 ms per series.
#
# Run this notebook NUM_SHARDS times, changing SHARD each run, then "Save
# Version" each run so each output becomes its own dataset.  Attach all shards
# to the training notebook.
# =============================================================================

# --- CELL 1 -----------------------------------------------------------------
# !pip install -q pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm

# Bootstrap.  Uses the step-00 output when it is attached, and fetches the
# modules itself when it is not - so this notebook runs standalone with internet
# ON, and off the attached output when internet is OFF.
import os, sys, glob, subprocess

REPO = "https://github.com/rahpalrah/ResumeParser"
BRANCH = "claude/knee-mri-abnormalities-kaggle-6jzvyk"

# CLONE FIRST, fall back to an attached copy only when there is no network.
#
# The earlier order (attached wins) had a trap: every step copies knee_*.py into
# /kaggle/working, so those files end up inside that notebook's saved output.
# Attaching an earlier step's output for its parquet then ALSO pinned the code
# to whatever it looked like that day, and later fixes were invisible. Cloning
# first means the only notebook running old code is the offline one, which
# cannot clone anyway.
def _attached_code():
    # Several depths: Kaggle mounts a dataset at /kaggle/input/<slug>/ or at
    # /kaggle/input/datasets/<owner>/<slug>/, and a notebook output deeper
    # still. A single-level glob finds nothing in the nested layout.
    hits = []
    for d in range(1, 5):
        hits += glob.glob("/kaggle/input/" + "*/" * d + "knee_common.py")
    return os.path.dirname(sorted(hits)[0]) if hits else None

CODE_DIR = None
try:
    subprocess.run(["rm", "-rf", "/kaggle/tmp/repo"], check=False)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO,
                    "/kaggle/tmp/repo"], check=True, timeout=120)
    subprocess.run("cp /kaggle/tmp/repo/rsna_knee/*.py /kaggle/working/",
                   shell=True, check=True)
    CODE_DIR = "/kaggle/working"
    print("cloned fresh from", BRANCH)
except Exception as e:
    CODE_DIR = _attached_code()
    print(f"clone unavailable ({type(e).__name__}); using attached code at {CODE_DIR}")
assert CODE_DIR and os.path.exists(os.path.join(CODE_DIR, "knee_common.py")), (
    "Could not obtain the modules. Either turn internet ON, or run step 00 and "
    "attach its output (Add Data -> Your Work -> Notebook Output).")

# Drop any already-imported copy: Python caches modules, so without this a
# re-run of this cell keeps the version imported earlier in the session.
for _m in [m for m in list(sys.modules) if m.startswith("knee_")]:
    del sys.modules[_m]
sys.path.insert(0, CODE_DIR)
print("code from:", CODE_DIR)

import time, math, json, traceback
import numpy as np, pandas as pd
from multiprocessing import Pool
import knee_common as kc

COMP = kc.find_comp_dir()          # autodetected: never hard-code the slug
print("competition data:", COMP)
OUT = "/kaggle/working"

SHARD = 0            # <-- CHANGE THIS each run: 0,1,2,...,NUM_SHARDS-1
NUM_SHARDS = 2       # step 0 measured ~1 h total on 4 procs, so 2 runs of ~25
                     # min each. Raise it only if step 0 projects over ~11 h.
SPLIT = "train"      # "train" or "test"
N_PROC = 4           # Kaggle CPU notebooks expose 4 vCPU

CFG = kc.Cfg(comp_dir=COMP, cache_dir=f"{OUT}/cache")
os.makedirs(CFG.cache_dir, exist_ok=True)

SERIES_ROOT = kc.find_series_root(COMP, SPLIT)
print("DICOM root:", SERIES_ROOT)
series = pd.read_csv(f"{COMP}/{SPLIT}_series.csv")
# Only series that win a slot are ever read by the model, so only those are
# decoded.  On a typical study that is 4 of 5-7 series - a third of the work
# disappears here before a single DICOM is touched.
series = pd.concat([kc.assign_slots(g) for _, g in series.groupby("StudyInstanceUID")])
series = series[series["slot"] >= 0].reset_index(drop=True)
print(f"{SPLIT}: {len(series):,} series in slots "
      f"(of {len(pd.read_csv(f'{COMP}/{SPLIT}_series.csv')):,} total)")

mine = series[np.arange(len(series)) % NUM_SHARDS == SHARD].reset_index(drop=True)
print(f"shard {SHARD}/{NUM_SHARDS}: {len(mine):,} series")

# --- CELL 2 -----------------------------------------------------------------
LAT_MAP = {"L": 0, "LEFT": 0, "R": 1, "RIGHT": 1}

def read_laterality(series_dir: str) -> int:
    """Knee side from the DICOM header; 2 when the tag was not in the
    allowlisted set for this site.  The model needs it to know which half of a
    coronal slice is the medial compartment."""
    import pydicom
    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))[:3]
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:
            continue
        for tag in ("ImageLaterality", "Laterality", "BodyPartExamined",
                    "SeriesDescription", "StudyDescription"):
            v = str(getattr(ds, tag, "") or "").strip().upper()
            if tag in ("ImageLaterality", "Laterality") and v in LAT_MAP:
                return LAT_MAP[v]
            if "LEFT" in v or v.endswith(" L"):
                return 0
            if "RIGHT" in v or v.endswith(" R"):
                return 1
    return 2


def work(rec: dict) -> dict:
    sdir = os.path.join(SERIES_ROOT, rec["StudyInstanceUID"],
                        rec["SeriesInstanceUID"])
    out = dict(rec); out["ok"] = 0; out["Laterality"] = 2; out["kb"] = 0
    dst = kc.sprite_path(CFG.cache_dir, rec["StudyInstanceUID"], rec["SeriesInstanceUID"])
    try:
        out["n_files"] = len(glob.glob(sdir + "/*.dcm"))
        if os.path.exists(dst):                      # resume after a timeout
            out["ok"] = 1
            out["Laterality"] = read_laterality(sdir)
            return out
        vol = kc.load_series_volume(sdir, CFG.n_slices, CFG.img_size)
        if vol is None:
            return out
        out["kb"] = kc.write_sprite(dst, vol, CFG.grid_w, CFG.jpeg_quality) / 1024
        out["Laterality"] = read_laterality(sdir)
        out["ok"] = 1
    except Exception:
        out["err"] = traceback.format_exc(limit=1)
    return out

# --- CELL 3 -----------------------------------------------------------------
recs = mine.to_dict("records")
t0 = time.time()
results = []
with Pool(N_PROC) as pool:
    for i, r in enumerate(pool.imap_unordered(work, recs, chunksize=4)):
        results.append(r)
        if (i + 1) % 200 == 0 or i + 1 == len(recs):
            el = time.time() - t0
            eta = el / (i + 1) * (len(recs) - i - 1)
            ok = sum(x["ok"] for x in results)
            print(f"  {i+1:>6}/{len(recs)}  ok={ok:>6}  "
                  f"{el/60:6.1f} min elapsed  ETA {eta/60:6.1f} min", flush=True)

meta = pd.DataFrame(results)
meta.to_parquet(f"{OUT}/series_meta_{SPLIT}_shard{SHARD}.parquet", index=False)
print(f"\nfailures: {int((meta['ok'] == 0).sum())}")
if (meta["ok"] == 0).any():
    print(meta[meta["ok"] == 0].head(5).to_string())
print(f"cache size this shard: {meta['kb'].sum()/2**20:.2f} GiB")
print("laterality distribution:", meta["Laterality"].value_counts().to_dict())

# --- CELL 4 -----------------------------------------------------------------
# Verify a random sprite decodes back to the right shape before you spend a
# "Save Version" on this shard.
row = meta[meta["ok"] == 1].sample(1).iloc[0]
vol = kc.read_sprite(kc.sprite_path(CFG.cache_dir, row.StudyInstanceUID,
                                    row.SeriesInstanceUID),
                     CFG.n_slices, CFG.img_size, CFG.grid_w)
print("round-trip:", vol.shape, vol.dtype, "min", vol.min(), "max", vol.max())

import matplotlib.pyplot as plt
plt.figure(figsize=(14, 3))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(vol[i * (CFG.n_slices // 8)], cmap="gray"); plt.axis("off")
plt.suptitle(f"{row.Anatomical_Plane} fluid={row.Fluid_Sensitive} "
             f"fat={row.Fat_Suppression} slot={row.slot}")
plt.show()
