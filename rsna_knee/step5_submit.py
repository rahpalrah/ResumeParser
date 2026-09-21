# =============================================================================
# STEP 5 - inference + submission.csv.             Kaggle notebook, GPU, 9 h cap.
# Internet: OFF (competition requirement).  Runtime: ~1.5 h for ~1300 studies.
#
# Attach: the competition data, your code dataset, your fold checkpoints, and a
# wheels dataset holding pylibjpeg / openjpeg / gdcm so the DICOM decoders can
# be installed with no network.
#
# There is no sprite cache at test time - series are decoded straight from
# DICOM inside the DataLoader workers, which is why the decode path in
# knee_common is written to touch only the slices it actually keeps.
# =============================================================================

# --- CELL 1 -----------------------------------------------------------------
# Run this in a cell of its own, FIRST (the wheels ride along with the code
# from step 00, so no network is needed):
#
#   import glob
#   W = glob.glob("/kaggle/input/*/wheels")[0]
#   !pip install -q --no-index --find-links={W} \
#       pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm
# Locate the step-00 bootstrap output.  Attached notebook outputs land under an
# unpredictable folder name, so find it by content rather than by name.
import os, sys, glob
_c = glob.glob("/kaggle/input/*/knee_common.py") + glob.glob("/kaggle/working/knee_common.py")
assert _c, "Attach the step-00 notebook output (Add Data -> Your Work -> Notebook Output)"
CODE_DIR = os.path.dirname(_c[0])
sys.path.insert(0, CODE_DIR)
print("code from:", CODE_DIR)

import time, math
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
import knee_common as kc
from knee_common import LABELS, N_SLOTS, PLANE2IDX, assign_slots
from knee_data import _to_25d, collate
from knee_model import CareNet

COMP = "/kaggle/input/rsna-knee-abnormalities-detection"
CKPTS = sorted(glob.glob("/kaggle/input/*/carenet_f*.pt"))
assert CKPTS, "Attach the step-4 notebook outputs (carenet_f*.pt)"
DEV = "cuda"
USE_TTA = True
T_BUDGET_S = 7.5 * 3600      # leave headroom inside the 9 h cap

test = pd.read_csv(f"{COMP}/test.csv")
tser = pd.read_csv(f"{COMP}/test_series.csv")
tser = pd.concat([assign_slots(g) for _, g in tser.groupby("StudyInstanceUID")])
print(f"{len(test):,} studies | {len(tser):,} series | {len(CKPTS)} checkpoints")

ck0 = torch.load(CKPTS[0], map_location="cpu", weights_only=False)
CFG = kc.Cfg(**{k: v for k, v in ck0["cfg"].items() if k in kc.Cfg.__dataclass_fields__})
CFG.pretrained = False                              # weights come from the ckpt
print("config:", CFG.backbone, CFG.img_size, CFG.n_slices)

# --- CELL 2 -----------------------------------------------------------------
LAT_MAP = {"L": 0, "LEFT": 0, "R": 1, "RIGHT": 1}

def read_laterality(series_dir):
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
            if "LEFT" in v:
                return 0
            if "RIGHT" in v:
                return 1
    return 2


class TestStudyDS(Dataset):
    """Same tensor contract as the training dataset, fed from raw DICOM."""

    def __init__(self, studies, series, cfg):
        self.uids = studies["StudyInstanceUID"].tolist()
        self.cfg = cfg
        s = series[series["slot"] >= 0]
        self.by_study = {u: g.to_dict("records") for u, g in s.groupby("StudyInstanceUID")}

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        cfg, uid = self.cfg, self.uids[i]
        recs = self.by_study.get(uid, [])
        S, K, Z = N_SLOTS, cfg.n_slices, cfg.img_size
        image = np.zeros((S, K, 3, Z, Z), np.float32)
        slot = np.arange(S, dtype=np.int64)
        plane = np.full(S, len(PLANE2IDX), np.int64)
        fluid = np.full(S, 2, np.int64)
        fat = np.full(S, 2, np.int64)
        mask = np.zeros(S, np.int64)
        lat = 2
        for s in range(S):
            cand = [r for r in recs if int(r["slot"]) == s]
            if not cand:
                continue
            rec = cand[0]
            sdir = os.path.join(COMP, "test_series", uid, rec["SeriesInstanceUID"])
            vol = kc.load_series_volume(sdir, K, Z)
            if vol is None:
                continue
            image[s] = _to_25d(vol)
            plane[s] = PLANE2IDX.get(str(rec.get("Anatomical_Plane", "")), len(PLANE2IDX))
            fluid[s] = int(rec["Fluid_Sensitive"]) if rec.get("Fluid_Sensitive") in (0, 1) else 2
            fat[s] = int(rec["Fat_Suppression"]) if rec.get("Fat_Suppression") in (0, 1) else 2
            mask[s] = 1
            if lat == 2:
                lat = read_laterality(sdir)
        return {"image": torch.from_numpy(image),
                "slot": torch.from_numpy(slot),
                "plane": torch.from_numpy(plane),
                "fluid": torch.from_numpy(fluid),
                "fat": torch.from_numpy(fat),
                "series_mask": torch.from_numpy(mask),
                "lat": torch.tensor(lat, dtype=torch.long),
                "uid": uid}


dl = DataLoader(TestStudyDS(test, tser, CFG), batch_size=4, shuffle=False,
                num_workers=4, pin_memory=True, collate_fn=collate)

# --- CELL 3 -----------------------------------------------------------------
t0 = time.time()
preds = []
for ci, path in enumerate(CKPTS):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = CareNet(CFG)
    model.load_state_dict(ck["model"])
    model = model.to(DEV).eval()
    P = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for bi, b in enumerate(dl):
            x = {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in b.items()}
            p = torch.sigmoid(model(x)["logits"]).float()
            if USE_TTA:
                xf = dict(x)
                xf["image"] = torch.flip(x["image"], dims=[-1])
                xf["lat"] = torch.where(x["lat"] < 2, 1 - x["lat"], x["lat"])
                p = 0.5 * (p + torch.sigmoid(model(xf)["logits"]).float())
            P.append(p.cpu().numpy())
            if bi % 50 == 0:
                print(f"  ckpt {ci} batch {bi}/{len(dl)} "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
    preds.append(np.concatenate(P))
    del model; torch.cuda.empty_cache()
    # stop adding folds if the remaining time would not fit another pass
    per_ckpt = (time.time() - t0) / (ci + 1)
    if time.time() - t0 + per_ckpt > T_BUDGET_S:
        print(f"time budget: stopping after {ci+1} checkpoints")
        break

# Rank-average across folds: AUC only cares about order, and rank averaging is
# immune to one fold being systematically over-confident.
P = np.mean([kc.rank_normalise(p) for p in preds], axis=0)

# --- CELL 4 -----------------------------------------------------------------
sub = pd.DataFrame(P, columns=LABELS)
sub.insert(0, "StudyInstanceUID", test["StudyInstanceUID"].values)
sub = sub[["StudyInstanceUID"] + LABELS]

template = pd.read_csv(f"{COMP}/sample_submission.csv")
assert list(sub.columns) == list(template.columns), (list(sub.columns), list(template.columns))
assert len(sub) == len(test) and sub["StudyInstanceUID"].is_unique
assert sub[LABELS].notna().all().all() and sub[LABELS].values.min() >= 0

sub.to_csv("submission.csv", index=False)
print(sub.head())
print(f"\nwrote submission.csv  {sub.shape}  in {(time.time()-t0)/60:.1f} min")
