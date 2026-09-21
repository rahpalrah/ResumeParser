# =============================================================================
# DIAGNOSTIC - what do the headers of the unknown-laterality studies contain?
#
# ~49% of studies expose no side through the tags read_laterality checks. That
# is either a recoverable parsing gap or a genuine property of the allowlisted
# 86 tags, and guessing which has already cost one wrong theory. This reads
# the headers and reports.
#
# Run it with the one-cell runner:  STEP = "diag_laterality.py"
# =============================================================================
import os, sys, glob, collections
import numpy as np, pandas as pd
import knee_common as kc
import pydicom

COMP = kc.find_comp_dir()
SERIES_ROOT = kc.find_series_root(COMP, "train")
N_STUDIES = 150

meta_paths = sorted(glob.glob("/kaggle/working/series_meta_train_shard*.parquet")) \
             or kc.find_inputs("series_meta_train_shard*.parquet")
assert meta_paths, "run step 3 first - this reads its shard manifest"
meta = pd.concat([pd.read_parquet(p) for p in meta_paths], ignore_index=True)
meta = meta[meta["ok"] == 1]

per_study = meta.groupby("StudyInstanceUID")["Laterality"].apply(
    lambda v: int(all(x == 2 for x in v)))
unknown = per_study[per_study == 1].index.tolist()
known = per_study[per_study == 0].index.tolist()
print(f"{len(unknown):,} studies with no side on any series, "
      f"{len(known):,} with one ({len(unknown)/max(1,len(per_study)):.1%} unknown)\n")

def one_header(study):
    row = meta[meta["StudyInstanceUID"] == study].iloc[0]
    d = os.path.join(SERIES_ROOT, study, row["SeriesInstanceUID"])
    f = sorted(glob.glob(d + "/*.dcm"))
    if not f:
        return None
    return pydicom.dcmread(f[0], stop_before_pixels=True, force=True)

# ---- 1. which tags exist at all, and how often they carry a side ------------
present = collections.Counter()
side_hint = collections.Counter()
values = collections.defaultdict(collections.Counter)
SIDE = ("LEFT", "RIGHT", "\bL\b", "\bR\b", "LT", "RT")
for study in unknown[:N_STUDIES]:
    ds = one_header(study)
    if ds is None:
        continue
    for el in ds:
        if el.VR in ("SQ", "OB", "OW", "UN") or el.keyword in ("PixelData",):
            continue
        v = str(el.value)[:60].strip()
        if not v:
            continue
        present[el.keyword or str(el.tag)] += 1
        values[el.keyword or str(el.tag)][v] += 1
        if any(h in v.upper() for h in ("LEFT", "RIGHT", "LT", "RT", " L", " R")):
            side_hint[el.keyword or str(el.tag)] += 1

print(f"tags present in the unknown-laterality headers (of {min(N_STUDIES, len(unknown))} sampled):")
for k, c in present.most_common():
    hint = side_hint.get(k, 0)
    flag = f"   <-- {hint} carry a possible side" if hint else ""
    print(f"  {k:<34s} {c:>4}{flag}")

# ---- 2. the free-text tags, verbatim ----------------------------------------
print("\nmost common values of the descriptive tags:")
for k in ("SeriesDescription", "StudyDescription", "ProtocolName", "BodyPartExamined",
          "PerformedProcedureStepDescription", "RequestedProcedureDescription",
          "ImageComments", "PatientPosition", "Laterality", "ImageLaterality"):
    if k not in values:
        continue
    print(f"\n  {k}:")
    for v, c in values[k].most_common(8):
        print(f"    {c:>4}  {v!r}")

# ---- 3. a full header, for contrast -----------------------------------------
if known:
    print("\n" + "=" * 66)
    print("a study that DOES resolve, for comparison:")
    ds = one_header(known[0])
    for el in ds:
        if el.VR not in ("SQ", "OB", "OW", "UN"):
            print(f"  {el.keyword or el.tag:<34s} {str(el.value)[:60]!r}")
