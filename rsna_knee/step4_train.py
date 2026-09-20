# =============================================================================
# STEP 4 - train CARE-Net.                         Kaggle notebook, GPU T4 x2.
# Internet: ON (timm weights).  Runtime: ~7 h for one fold. Run once per fold.
#
# Two stages, which is the whole point of the pipeline:
#   A) distillation pass over EVERY study, against the report teacher's soft
#      targets.  This is where the model learns what a torn ACL looks like,
#      using ~10x more studies than the gold labels alone provide.
#   B) fine-tune on the gold-labelled studies only, at a lower LR.
# Validation is always gold-only, on the held-out fold.
# =============================================================================

# --- CELL 1 -----------------------------------------------------------------
# !pip install -q timm iterative-stratification
import os, sys, gc, math, time, json
sys.path.insert(0, "/kaggle/working")
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
import knee_common as kc
from knee_data import KneeStudyDataset, collate, make_folds
from knee_model import CareNet, SoftAsymmetricLoss, ModelEMA

COMP = "/kaggle/input/rsna-knee-abnormalities-detection"
OUT = "/kaggle/working"
# every step-3 shard you saved, attached as a dataset
CACHE_DIRS = sorted(__import__("glob").glob("/kaggle/input/*/cache"))
META_GLOB = "/kaggle/input/*/series_meta_train_shard*.parquet"
TARGETS = "/kaggle/input/knee-text-teacher/study_targets.parquet"

FOLD = 0                       # <-- CHANGE per run
EFFICIENCY = False             # True -> the small config for the efficiency track
CFG = (kc.CFG_EFFICIENCY if EFFICIENCY else kc.Cfg())
CFG.comp_dir, CFG.out_dir = COMP, OUT
kc.seed_everything(CFG.seed + FOLD)
DEV = "cuda"

# --- CELL 2 -----------------------------------------------------------------
import glob as _glob
meta = pd.concat([pd.read_parquet(p) for p in sorted(_glob.glob(META_GLOB))],
                 ignore_index=True)
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
_orig_sprite_path = kc.sprite_path
kc.sprite_path = lambda cache_dir, s, se: _mc.resolve(s, se)
import knee_data as kd
kd.sprite_path = kc.sprite_path
print("cache roots:", _mc.roots)

targets = pd.read_parquet(TARGETS)
targets = targets[targets["StudyInstanceUID"].isin(set(meta["StudyInstanceUID"]))].reset_index(drop=True)
gold = targets[targets["is_gold"] == 1].reset_index(drop=True)
soft = targets[targets["is_gold"] == 0].reset_index(drop=True)
print(f"gold {len(gold):,} | soft {len(soft):,}")

folds = make_folds(gold, CFG.folds, seed=CFG.seed)
tr_gold = gold[folds != FOLD].reset_index(drop=True)
va_gold = gold[folds == FOLD].reset_index(drop=True)
print(f"fold {FOLD}: train {len(tr_gold):,} / valid {len(va_gold):,}")

def make_loader(df, train, weight=1.0, bs=None):
    ds = KneeStudyDataset(df, meta, CFG, train=train,
                          targets=df[kc.LABELS].values.astype(np.float32),
                          weights=np.full(len(df), weight, np.float32))
    return DataLoader(ds, batch_size=bs or CFG.batch_size, shuffle=train,
                      num_workers=CFG.num_workers, pin_memory=True,
                      drop_last=train, collate_fn=collate,
                      persistent_workers=CFG.num_workers > 0)

# stage A sees the soft studies plus the in-fold gold studies
stage_a_df = pd.concat([soft, tr_gold], ignore_index=True)
stage_a_w = np.concatenate([np.full(len(soft), CFG.pseudo_weight, np.float32),
                            np.ones(len(tr_gold), np.float32)])
dl_a = DataLoader(
    KneeStudyDataset(stage_a_df, meta, CFG, train=True,
                     targets=stage_a_df[kc.LABELS].values.astype(np.float32),
                     weights=stage_a_w),
    batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers,
    pin_memory=True, drop_last=True, collate_fn=collate)
dl_b = make_loader(tr_gold, True)
dl_va = make_loader(va_gold, False, bs=CFG.batch_size * 2)

# --- CELL 3 -----------------------------------------------------------------
from knee_model import MEDIAL_IDX, LATERAL_IDX

model = CareNet(CFG).to(DEV)
n_par = sum(p.numel() for p in model.parameters()) / 1e6
print(f"CARE-Net {CFG.backbone}: {n_par:.1f} M parameters")

lossf = SoftAsymmetricLoss()
ema = ModelEMA(model, CFG.ema_decay)
scaler = torch.cuda.amp.GradScaler()

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
        with torch.cuda.amp.autocast():
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
            with torch.cuda.amp.autocast():
                o = model(x)
                loss = lossf(o["logits"], x["target"], x["weight"])
                if "logit_med" in o:   # deep supervision on the compartment branch
                    loss = loss + 0.3 * (
                        lossf(o["logit_med"], x["target"][:, MEDIAL_IDX], x["weight"]) +
                        lossf(o["logit_lat"], x["target"][:, LATERAL_IDX], x["weight"]))
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
        print(f"  [{name}] epoch {ep}: loss {run/max(len(loader),1):.4f} "
              f"| val macro AUC {m:.5f} | {(time.time()-t0)/60:.1f} min")
        best = max(best, m)
        if m > GLOBAL_BEST:
            GLOBAL_BEST = m
            torch.save({"model": ema.ema.state_dict(), "cfg": CFG.__dict__,
                        "fold": FOLD, "auc": m},
                       f"{OUT}/carenet_f{FOLD}.pt")
            print(f"    saved (best so far {GLOBAL_BEST:.5f})")
    return best

print("\n=== STAGE A: report distillation over all studies ===")
best_a = train_stage("A", dl_a, CFG.epochs_pseudo)
print("\n=== STAGE B: gold fine-tune ===")
best_b = train_stage("B", dl_b, CFG.epochs_gold, lr_scale=0.3)

# --- CELL 4 -----------------------------------------------------------------
ck = torch.load(f"{OUT}/carenet_f{FOLD}.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
y, p = evaluate(model, dl_va, tta=True)
m, per = kc.macro_auc(y, p)
print(f"\nFOLD {FOLD} final (with mirror TTA): {m:.5f}")
kc.print_auc_table(m, per)

np.save(f"{OUT}/oof_pred_f{FOLD}.npy", p)
va_gold[["StudyInstanceUID"]].to_csv(f"{OUT}/oof_uid_f{FOLD}.csv", index=False)
json.dump({"fold": FOLD, "macro_auc": m, "per_label": per,
           "stage_a": best_a, "stage_b": best_b},
          open(f"{OUT}/fold{FOLD}_score.json", "w"), indent=2)
print("saved carenet_f%d.pt + oof" % FOLD)
