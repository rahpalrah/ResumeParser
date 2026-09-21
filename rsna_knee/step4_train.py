# =============================================================================
# STEP 4 - train CARE-Net.                         Kaggle notebook, GPU T4 x2.
# Internet: ON (timm weights).  Runtime: ~7 h for one fold. Run once per fold.
#
# The gold labels cover ~58 studies. That is too few to fine-tune on and far
# too few to validate on - 12 studies per fold would put 2 positives in the MCL
# column, and a macro AUC computed on that is noise.
#
# So the split is:
#   TRAIN  on the report teacher's targets, over every study with no gold label.
#   SELECT on a held-out fold of those same teacher targets (n~430). Noisy in
#          absolute terms, but large enough to rank checkpoints.
#   REPORT on the gold studies, held out of every fit, scored per cell with a
#          mask. This is the honest estimate, and it is a sanity check at that
#          sample size - never tune against it.
#
# Per-cell weights let a real annotation outweigh a teacher guess wherever gold
# exists.
# =============================================================================

# --- CELL 1 -----------------------------------------------------------------
import subprocess, sys
for _mod, _pkg in [("timm", "timm"), ("iterstrat", "iterative-stratification")]:
    try:
        __import__(_mod)
    except Exception:
        print("installing", _pkg, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)
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

import gc, math, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
import knee_common as kc
from knee_data import KneeStudyDataset, collate, make_folds
from knee_model import CareNet, SoftAsymmetricLoss, ModelEMA

COMP = kc.find_comp_dir()          # autodetected: never hard-code the slug
print("competition data:", COMP)
OUT = "/kaggle/working"
# Every attached step-3 shard, plus the step-2 targets.  All located by content
# so the notebook does not care what the attached outputs are named.
CACHE_DIRS = [d for d in kc.find_inputs("cache") if os.path.isdir(d)]
META_SHARDS = kc.find_inputs("series_meta_train_shard*.parquet")
_t = kc.find_inputs("study_targets.parquet")
assert _t, "Attach the step-2 notebook output (study_targets.parquet)"
TARGETS = _t[0]

FOLD = globals().get("FOLD", 0)
EFFICIENCY = globals().get("EFFICIENCY", False)   # True -> efficiency-track config
# Kaggle stops a GPU session at 9 h and the checkpoint dies with it. Training
# stops itself with time to spare and keeps the best epoch so far, which beats
# discovering the limit the hard way five hours in.
TIME_BUDGET_H = globals().get("TIME_BUDGET_H", 7.5)
T_START = time.time()
CFG = (kc.CFG_EFFICIENCY if EFFICIENCY else kc.Cfg())
CFG.comp_dir, CFG.out_dir = COMP, OUT
kc.seed_everything(CFG.seed + FOLD)
DEV = "cuda"

# --- CELL 2 -----------------------------------------------------------------
_shards = sorted(META_SHARDS)
assert _shards, "Attach the step-3 shard outputs"
print(f"{len(_shards)} shard manifests")
meta = pd.concat([pd.read_parquet(p) for p in _shards], ignore_index=True)
meta = meta[meta["ok"] == 1].drop_duplicates("SeriesInstanceUID").reset_index(drop=True)
print(f"cached series: {len(meta):,} over {meta.StudyInstanceUID.nunique():,} studies")

# The cache is spread over several attached datasets; resolve each sprite once.
class MultiCache:
    """Presents several read-only cache roots as one directory to the dataset."""
    def __init__(self, roots): self.roots = roots
    def resolve(self, study, series):
        for r in self.roots:
            p = kc.sprite_path(r, study, series)
            if os.path.exists(p):
                return p
        return kc.sprite_path(self.roots[0], study, series)

_mc = MultiCache(CACHE_DIRS if CACHE_DIRS else [f"{OUT}/cache"])
kc.sprite_path = lambda cache_dir, s, se: _mc.resolve(s, se)
import knee_data as kd
kd.sprite_path = kc.sprite_path
print("cache roots:", _mc.roots)

# The DICOM headers of ~48% of studies carry no Laterality tag - the
# diagnostic confirmed it is absent from the allowlisted set, not empty. The
# reports name the side, so recover it for TRAINING. The test set has no
# reports and keeps whatever its headers give; that is not a mismatch, because
# the compartment branch is gated per study and simply falls back to the main
# head when the side is unknown. More sides at training time means the branch
# is better trained for the test studies that do have one.
from knee_text import norm_text, laterality_from_report
_rep = pd.read_csv(f"{COMP}/train.csv", usecols=["StudyInstanceUID", "Report"])
_rep["lat_report"] = _rep["Report"].map(lambda t: laterality_from_report(norm_text(t)))
_lat = dict(zip(_rep["StudyInstanceUID"], _rep["lat_report"]))

_before = meta.groupby("StudyInstanceUID")["Laterality"].apply(
    lambda v: int(all(x == 2 for x in v))).mean()
_fill = meta["StudyInstanceUID"].map(_lat).fillna(2).astype(int)
meta["Laterality"] = np.where(meta["Laterality"] == 2, _fill, meta["Laterality"])
_after = meta.groupby("StudyInstanceUID")["Laterality"].apply(
    lambda v: int(all(x == 2 for x in v))).mean()
print(f"laterality unknown: {_before:.1%} of studies from DICOM -> {_after:.1%} "
      f"after reading the reports")

targets = pd.read_parquet(TARGETS)
targets = targets[targets["StudyInstanceUID"].isin(set(meta["StudyInstanceUID"]))]
targets = targets.reset_index(drop=True)

GOLD_COLS = [f"is_gold_{c}" for c in kc.LABELS]
assert set(GOLD_COLS) <= set(targets.columns), (
    "study_targets.parquet has no per-cell gold flags - re-run step 2.")
GOLD_M = targets[GOLD_COLS].values.astype(np.float32)
is_gold_row = GOLD_M.max(axis=1) > 0

gold_df = targets[is_gold_row].reset_index(drop=True)
pseudo_df = targets[~is_gold_row].reset_index(drop=True)
gold_mask = GOLD_M[is_gold_row]
print(f"gold holdout {len(gold_df):,} studies ({int(gold_mask.sum()):,} annotated cells) "
      f"| teacher-labelled {len(pseudo_df):,}")

folds = make_folds(pseudo_df, CFG.folds, seed=CFG.seed)
tr_df = pseudo_df[folds != FOLD].reset_index(drop=True)
va_df = pseudo_df[folds == FOLD].reset_index(drop=True)
print(f"fold {FOLD}: train {len(tr_df):,} / select-on {len(va_df):,} / gold {len(gold_df):,}")

def loader(df, train, weights=None, bs=None):
    ds = KneeStudyDataset(df, meta, CFG, train=train,
                          targets=df[kc.LABELS].values.astype(np.float32),
                          weights=weights)
    return DataLoader(ds, batch_size=bs or CFG.batch_size, shuffle=train,
                      num_workers=CFG.num_workers, pin_memory=True,
                      drop_last=train, collate_fn=collate)

dl_tr = loader(tr_df, True, np.full((len(tr_df), kc.N_LABELS), 1.0, np.float32))
dl_va = loader(va_df, False, bs=CFG.batch_size * 2)
dl_gold = loader(gold_df, False, bs=CFG.batch_size * 2)

# --- CELL 3 -----------------------------------------------------------------
from knee_model import MEDIAL_IDX, LATERAL_IDX

model = CareNet(CFG).to(DEV)
n_par = sum(p.numel() for p in model.parameters()) / 1e6
print(f"CARE-Net {CFG.backbone}: {n_par:.1f} M parameters")

lossf = SoftAsymmetricLoss()
ema = ModelEMA(model, CFG.ema_decay)
scaler = kc.make_grad_scaler()

def param_groups(m):
    bb, hd = [], []
    for n, p in m.named_parameters():
        (bb if n.startswith("backbone.") else hd).append(p)
    return [{"params": bb, "lr": CFG.lr_backbone},
            {"params": hd, "lr": CFG.lr_head}]

@torch.no_grad()
def evaluate(net, loader, tta=False):
    net.eval()
    P, Y = [], []
    for b in loader:
        x = {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v)
             for k, v in b.items()}
        with kc.amp_autocast():
            p = torch.sigmoid(net(x)["logits"]).float()
            if tta:   # mirror TTA, with the laterality token flipped to match
                xf = dict(x)
                xf["image"] = torch.flip(x["image"], dims=[-1])
                xf["lat"] = torch.where(x["lat"] < 2, 1 - x["lat"], x["lat"])
                p = 0.5 * (p + torch.sigmoid(net(xf)["logits"]).float())
        P.append(p.cpu().numpy()); Y.append(b["target"].numpy())
    return np.concatenate(Y), np.concatenate(P)

GLOBAL_BEST = -1.0

def train_stage(name, loader, epochs, lr_scale=1.0):
    # `global` so stage B can never overwrite a better stage-A checkpoint with
    # a worse one - the two stages share one output file on purpose.
    global GLOBAL_BEST
    opt = torch.optim.AdamW(param_groups(model), weight_decay=CFG.weight_decay)
    for g in opt.param_groups:
        g["lr"] *= lr_scale
    total = max(1, len(loader) // CFG.accum) * epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=total, pct_start=0.1)
    best = -1.0
    for ep in range(epochs):
        model.train(); t0 = time.time(); run = 0.0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(loader):
            x = {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in b.items()}
            with kc.amp_autocast():
                o = model(x)
                loss = lossf(o["logits"], x["target"], x["weight"])
                if "logit_med" in o:   # deep supervision on the compartment branch
                    # Only on studies whose side is known: the halves carry no
                    # compartment meaning otherwise, and supervising them there
                    # teaches the medial head lateral anatomy.
                    cw = o["comp_valid"].unsqueeze(-1)
                    w = x["weight"] if x["weight"].dim() == 2 else x["weight"].view(-1, 1)
                    loss = loss + 0.3 * (
                        lossf(o["logit_med"], x["target"][:, MEDIAL_IDX],
                              w[:, :1] * cw) +
                        lossf(o["logit_lat"], x["target"][:, LATERAL_IDX],
                              w[:, :1] * cw))
                loss = loss / CFG.accum
            scaler.scale(loss).backward()
            run += loss.item() * CFG.accum
            if (i + 1) % CFG.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                if sch.last_epoch < total - 1:
                    sch.step()
                ema.update(model)
            if (i + 1) % 200 == 0:
                print(f"    {name} ep{ep} {i+1}/{len(loader)} "
                      f"loss {run/(i+1):.4f} {(time.time()-t0)/60:.1f} min", flush=True)
        y, p = evaluate(ema.ema, dl_va)
        m, per = kc.macro_auc(y, p)
        yg, pg = evaluate(ema.ema, dl_gold)
        m_gold, _ = kc.macro_auc(yg, pg, mask=gold_mask)
        print(f"  [{name}] epoch {ep}: loss {run/max(len(loader),1):.4f} "
              f"| select AUC {m:.5f} | gold AUC {m_gold:.5f} "
              f"| {(time.time()-t0)/60:.1f} min")
        spent = (time.time() - T_START) / 3600
        per_epoch = spent / (ep + 1)
        if spent + per_epoch > TIME_BUDGET_H and ep + 1 < epochs:
            print(f"    stopping after epoch {ep}: {spent:.1f} h spent, another "
                  f"{per_epoch:.1f} h would pass the {TIME_BUDGET_H} h budget")
            _stop = True
        else:
            _stop = False
        best = max(best, m)
        if m > GLOBAL_BEST:
            GLOBAL_BEST = m
            torch.save({"model": ema.ema.state_dict(), "cfg": CFG.__dict__,
                        "fold": FOLD, "auc": m},
                       f"{OUT}/carenet_f{FOLD}.pt")
            print(f"    saved (best so far {GLOBAL_BEST:.5f})")
        if _stop:
            break
    return best

EPOCHS = globals().get("EPOCHS", CFG.epochs_pseudo + CFG.epochs_gold)
print(f"\n=== training on the report teacher's targets: {EPOCHS} epochs, "
      f"{TIME_BUDGET_H} h budget ===")
best = train_stage("distil", dl_tr, EPOCHS)

# --- CELL 4 -----------------------------------------------------------------
ck = torch.load(f"{OUT}/carenet_f{FOLD}.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])

y, p = evaluate(model, dl_va, tta=True)
m, per = kc.macro_auc(y, p)
print(f"\nFOLD {FOLD} against the teacher targets, mirror TTA: {m:.5f}")
print("  (how well it reproduces the teacher - the selection signal)")

yg, pg = evaluate(model, dl_gold, tta=True)
m_gold, per_gold = kc.macro_auc(yg, pg, mask=gold_mask)
print(f"\nFOLD {FOLD} against the {len(gold_df)} GOLD studies, mirror TTA:")
kc.print_auc_table(m_gold, per_gold)
print(f"  n={len(gold_df)}: wide error bars. A label showing nan had too few "
      f"annotated rows or only one class.")

np.save(f"{OUT}/oof_pred_f{FOLD}.npy", p)
va_df[["StudyInstanceUID"]].to_csv(f"{OUT}/oof_uid_f{FOLD}.csv", index=False)
np.save(f"{OUT}/gold_pred_f{FOLD}.npy", pg)
json.dump({"fold": FOLD, "select_auc": m, "gold_auc": m_gold,
           "per_label_gold": per_gold, "best_select": best},
          open(f"{OUT}/fold{FOLD}_score.json", "w"), indent=2)
print(f"saved carenet_f{FOLD}.pt + oof")
