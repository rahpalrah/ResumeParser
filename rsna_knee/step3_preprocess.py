# =============================================================================
# STEP 3 - build the sprite cache.          ONE CELL. Kaggle notebook, CPU.
# Internet: ON (to install the DICOM decoders).   Runtime: ~45 min, measured.
#
# 570 GB of DICOM cannot be read once per epoch. This reads every series that
# wins a slot exactly once and writes it as ONE grayscale JPEG "sprite sheet"
# holding the sampled slices in a grid, collapsing the training set to ~5 GiB
# that loads in ~2 ms per series.
#
# Everything is automatic: it installs the decoders, fetches the modules,
# measures its own throughput on a probe batch, decides whether one shard fits
# the time budget, aborts early if nothing is decoding, and checks the quality
# gates itself. The only manual step left is Save Version, which is a UI
# action - the last line reminds you.
# =============================================================================

import os, sys, glob, math, time, json, shutil, traceback, subprocess
from multiprocessing import Pool

# Settings read from globals() first, so the one-cell runner can set them
# without editing this file - `SHARD = 1` before the exec is enough.
SPLIT = globals().get("SPLIT", "train")   # "test" is never needed: step 5
                                          # decodes DICOM directly
SHARD = globals().get("SHARD", 0)         # only if the probe says to shard
NUM_SHARDS = globals().get("NUM_SHARDS", None)   # None = decide from throughput
TIME_BUDGET_H = globals().get("TIME_BUDGET_H", 9.0)  # inside the 12 h CPU limit
N_PROBE = 48          # series decoded to measure throughput (kept, not wasted)
REPO = "https://github.com/rahpalrah/ResumeParser"
BRANCH = "claude/knee-mri-abnormalities-kaggle-6jzvyk"

# ---- 1. decoders ------------------------------------------------------------
# JPEG Lossless and JPEG 2000 both appear in this corpus and neither decodes
# without a plugin. Installing quietly is cheaper than a 45 min run of zeros.
_need = []
for mod, pkg in [("libjpeg", "pylibjpeg-libjpeg"), ("openjpeg", "pylibjpeg-openjpeg"),
                 ("gdcm", "python-gdcm"), ("pylibjpeg", "pylibjpeg")]:
    try:
        __import__(mod)
    except Exception:
        _need.append(pkg)
if _need:
    print("installing DICOM decoders:", " ".join(_need), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_need], check=True)
else:
    print("DICOM decoders already present")

# ---- 2. modules -------------------------------------------------------------
def _attached_code():
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
    "Could not obtain the modules. Turn internet ON, or run step 00 and attach it.")
for _m in [m for m in list(sys.modules) if m.startswith("knee_")]:
    del sys.modules[_m]
sys.path.insert(0, CODE_DIR)

import numpy as np, pandas as pd
import knee_common as kc

print("code from:", CODE_DIR)
print("decoder status:")
_st = kc.check_dicom_backends()
assert _st.get("pydicom"), "pydicom missing - nothing can be read"
if not any(_st.get(k) for k in ("libjpeg", "openjpeg", "gdcm")):
    raise RuntimeError(
        "No JPEG decoder available. Restart the kernel (the plugins were just "
        "installed into a running session) and run this cell again.")

COMP = kc.find_comp_dir()
OUT = "/kaggle/working"
N_PROC = max(1, (os.cpu_count() or 4))
CFG = kc.Cfg(comp_dir=COMP, cache_dir=f"{OUT}/cache")
os.makedirs(CFG.cache_dir, exist_ok=True)
SERIES_ROOT = kc.find_series_root(COMP, SPLIT)
print(f"competition data: {COMP}\nDICOM root: {SERIES_ROOT}\nworkers: {N_PROC}")

# ---- 3. which series ---------------------------------------------------------
_all = pd.read_csv(f"{COMP}/{SPLIT}_series.csv")
series = pd.concat([kc.assign_slots(g) for _, g in _all.groupby("StudyInstanceUID")])
series = series[series["slot"] >= 0].reset_index(drop=True)
print(f"\n{SPLIT}: {len(series):,} series win a slot, of {len(_all):,} total "
      f"({1 - len(series)/len(_all):.0%} skipped before a DICOM is touched)")

# ---- 4. the worker -----------------------------------------------------------
LAT_MAP = {"L": 0, "LEFT": 0, "R": 1, "RIGHT": 1}

def read_laterality(series_dir):
    """Knee side from the header; 2 when the tag was not allowlisted for this
    site. The model needs it to know which half of a coronal slice is medial."""
    import pydicom
    for f in sorted(glob.glob(os.path.join(series_dir, "*.dcm")))[:3]:
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


def work(rec):
    sdir = os.path.join(SERIES_ROOT, rec["StudyInstanceUID"], rec["SeriesInstanceUID"])
    out = dict(rec); out["ok"] = 0; out["Laterality"] = 2; out["kb"] = 0.0
    out["std"] = 0.0
    dst = kc.sprite_path(CFG.cache_dir, rec["StudyInstanceUID"], rec["SeriesInstanceUID"])
    try:
        out["n_files"] = len(glob.glob(sdir + "/*.dcm"))
        if os.path.exists(dst):                       # resume within a session
            back = kc.read_sprite(dst, CFG.n_slices, CFG.img_size, CFG.grid_w)
            out["ok"] = 1
            out["kb"] = os.path.getsize(dst) / 1024
            out["std"] = float(back.std()) if back is not None else 0.0
            out["Laterality"] = read_laterality(sdir)
            return out
        vol = kc.load_series_volume(sdir, CFG.n_slices, CFG.img_size)
        if vol is None:
            return out
        out["kb"] = kc.write_sprite(dst, vol, CFG.grid_w, CFG.jpeg_quality) / 1024
        out["std"] = float(vol.std())
        out["Laterality"] = read_laterality(sdir)
        out["ok"] = 1
    except Exception:
        out["err"] = traceback.format_exc(limit=1)
    return out


def run_pool(records, label, report_every=200):
    t0, done = time.time(), []
    with Pool(N_PROC) as pool:
        for i, r in enumerate(pool.imap_unordered(work, records, chunksize=4)):
            done.append(r)
            n = i + 1
            if n == report_every and sum(x["ok"] for x in done) == 0:
                for x in done:
                    if x.get("err"):
                        print(x["err"])
                        break
                raise RuntimeError(
                    f"{report_every} series in and not one decoded. Something is "
                    f"wrong with the decoders or the DICOM root - aborting rather "
                    f"than burning the full run.")
            if n % report_every == 0 or n == len(records):
                el = time.time() - t0
                print(f"  [{label}] {n:>6}/{len(records)}  ok={sum(x['ok'] for x in done):>6}  "
                      f"{el/60:6.1f} min  ETA {el/n*(len(records)-n)/60:6.1f} min", flush=True)
    return done, time.time() - t0


# ---- 5. probe, then decide how much fits ------------------------------------
probe_recs = series.sample(min(N_PROBE, len(series)), random_state=0).to_dict("records")
print(f"\nprobing {len(probe_recs)} series to measure throughput...")
probe, probe_s = run_pool(probe_recs, "probe", report_every=len(probe_recs))
ok_probe = sum(x["ok"] for x in probe)
assert ok_probe, "no probe series decoded - see the traceback above"
per_series = probe_s / len(probe_recs)
total_h = per_series * len(series) / 3600
need_shards = max(1, math.ceil(total_h / TIME_BUDGET_H))
if NUM_SHARDS is None:
    NUM_SHARDS = need_shards
print(f"  {ok_probe}/{len(probe_recs)} decoded, {per_series:.2f} s/series wall "
      f"-> {total_h:.1f} h for all {len(series):,}")
print(f"  time budget {TIME_BUDGET_H} h -> NUM_SHARDS = {NUM_SHARDS}"
      + ("" if NUM_SHARDS == 1 else
         f"; run this cell once per SHARD in 0..{NUM_SHARDS-1}, Save Version each time"))

mine = series[np.arange(len(series)) % NUM_SHARDS == SHARD].reset_index(drop=True)
print(f"\nshard {SHARD}/{NUM_SHARDS}: {len(mine):,} series to process")

# ---- 6. the run --------------------------------------------------------------
results, elapsed = run_pool(mine.to_dict("records"), f"shard{SHARD}")
meta = pd.DataFrame(results)
meta_path = f"{OUT}/series_meta_{SPLIT}_shard{SHARD}.parquet"
meta.to_parquet(meta_path, index=False)

# ---- 7. the gates, checked here rather than by eye --------------------------
n = len(meta)
fail_rate = float((meta["ok"] == 0).mean())
unknown_lat = float((meta["Laterality"] == 2).mean())
gib = float(meta["kb"].sum()) / 2**20
# A near-flat sprite is a series that "succeeded" into blackness. One bad slice
# is replaced by its neighbour, which is fine; a whole series of them is a
# decoder that quietly gave up, and nothing else here would notice.
blank = float((meta.loc[meta["ok"] == 1, "std"] < 1.0).mean()) if (meta["ok"] == 1).any() else 1.0
print(f"\n{'='*66}\nshard {SHARD} finished in {elapsed/60:.1f} min -> {meta_path}")
gates = [
    ("decode failures < 2%", fail_rate < 0.02, f"{fail_rate:.2%} ({int(fail_rate*n)} of {n:,})"),
    ("unknown laterality < 30%", unknown_lat < 0.30, f"{unknown_lat:.1%}"),
    ("cache size < 15 GiB", gib < 15.0, f"{gib:.2f} GiB"),
    ("near-black sprites < 2%", blank < 0.02, f"{blank:.2%}"),
]
for name, passed, detail in gates:
    print(f"  {'PASS' if passed else 'FAIL'}  {name:<26s} {detail}")
if fail_rate >= 0.02:
    print("\nfirst failures:")
    print(meta[meta["ok"] == 0].head(3).to_string())
assert all(p for _, p, _ in gates), "a gate failed - read the rows above before saving"

# ---- 8. round trip, and look at it -------------------------------------------
row = meta[meta["ok"] == 1].sample(1, random_state=0).iloc[0]
vol = kc.read_sprite(kc.sprite_path(CFG.cache_dir, row.StudyInstanceUID,
                                    row.SeriesInstanceUID),
                     CFG.n_slices, CFG.img_size, CFG.grid_w)
assert vol is not None and vol.shape == (CFG.n_slices, CFG.img_size, CFG.img_size)
print(f"  PASS  sprite round-trip        {vol.shape}, range [{vol.min()}, {vol.max()}]")

import matplotlib.pyplot as plt
step = max(1, CFG.n_slices // 8)
plt.figure(figsize=(14, 3))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(vol[min(i * step, CFG.n_slices - 1)], cmap="gray"); plt.axis("off")
plt.suptitle(f"{row.Anatomical_Plane} fluid={row.Fluid_Sensitive} "
             f"fat={row.Fat_Suppression} slot={row.slot} - must look like knee MRI")
plt.show()

print(f"\n{'='*66}")
print("ALL GATES PASSED.  Now: Save Version -> Quick Save.")
print("Quick Save keeps /kaggle/working as it stands; Save & Run All would")
print(f"re-run the whole {elapsed/60:.0f} minutes for nothing.")
if NUM_SHARDS > 1:
    nxt = SHARD + 1
    print(f"Then set SHARD = {nxt} and repeat"
          if nxt < NUM_SHARDS else "That was the last shard.")
