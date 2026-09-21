# =============================================================================
# STEP 2 - the report teacher.                     Kaggle notebook, GPU T4 x2.
# Internet: ON (downloads xlm-roberta-base).  Runtime: ~1.5 h for 5 folds.
#
# Why this step exists: the image model needs a target for the ~90% of training
# studies that have a report but no labels.  A multilingual text model fitted on
# the labelled subset reaches far higher AUC on the reports than any image model
# ever will on the pixels, so its predictions are a much better teacher signal
# than nothing.  The test set has no reports - this model never runs at
# inference, it only manufactures training targets.  That is the distillation
# half of CARE-Net.
# =============================================================================

# --- CELL 1 -----------------------------------------------------------------
# !pip install -q iterative-stratification
# Locate the step-00 bootstrap output.  Attached notebook outputs land under an
# unpredictable folder name, so find it by content rather than by name.
import os, sys, glob
_c = glob.glob("/kaggle/input/*/knee_common.py") + glob.glob("/kaggle/working/knee_common.py")
assert _c, "Attach the step-00 notebook output (Add Data -> Your Work -> Notebook Output)"
CODE_DIR = os.path.dirname(_c[0])
sys.path.insert(0, CODE_DIR)
print("code from:", CODE_DIR)

import gc, math, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
import knee_common as kc

COMP = "/kaggle/input/rsna-knee-abnormalities-detection"
OUT = "/kaggle/working"
MODEL_NAME = "xlm-roberta-base"     # multilingual by construction; the reports
                                    # are in several languages
MAX_LEN = 512
FOLDS = 5
EPOCHS = 3
BS = 8
LR = 2e-5
DEV = "cuda"
kc.seed_everything(42)

train = pd.read_csv(f"{COMP}/train.csv")
_r = glob.glob("/kaggle/input/*/rule_features.parquet") + [f"{OUT}/rule_features.parquet"]
_r = [p for p in _r if os.path.exists(p)]
assert _r, "Attach the step-1 notebook output (rule_features.parquet)"
rules = pd.read_parquet(_r[0])
train = train.merge(rules, on="StudyInstanceUID", how="left")
rule_cols = [f"rule_{c}" for c in kc.LABELS]
train[rule_cols] = train[rule_cols].fillna(0.0)

is_lab = train[kc.LABELS].notna().all(axis=1).values
lab_df = train[is_lab].reset_index(drop=True)
print(f"labelled {len(lab_df):,} | unlabelled {int((~is_lab).sum()):,}")

# --- CELL 2 -----------------------------------------------------------------
class ReportDS(Dataset):
    def __init__(self, df, tok, with_target=True):
        self.txt = df["Report"].fillna("").astype(str).tolist()
        self.rule = df[rule_cols].values.astype(np.float32)
        self.y = df[kc.LABELS].values.astype(np.float32) if with_target else None
        self.tok = tok

    def __len__(self): return len(self.txt)

    def __getitem__(self, i):
        enc = self.tok(self.txt[i], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        item = {k: v[0] for k, v in enc.items()}
        item["rule"] = torch.from_numpy(self.rule[i])
        if self.y is not None:
            item["target"] = torch.from_numpy(self.y[i])
        return item


class ReportTeacher(nn.Module):
    """Encoder + mean-pool, concatenated with the rule hits from step 1.

    Feeding the rules in explicitly is not redundant with the encoder: it gives
    the head a calibrated, language-agnostic prior for the rare labels where 512
    tokens of training text are not enough to learn the phrasing from scratch."""

    def __init__(self, name=MODEL_NAME):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h + kc.N_LABELS, kc.N_LABELS)

    def forward(self, input_ids, attention_mask, rule):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(torch.cat([self.drop(pooled), rule], dim=-1))


def run_fold(fold, folds, tok):
    tr = lab_df[folds != fold].reset_index(drop=True)
    va = lab_df[folds == fold].reset_index(drop=True)
    dl_tr = DataLoader(ReportDS(tr, tok), batch_size=BS, shuffle=True,
                       num_workers=2, drop_last=True, pin_memory=True)
    dl_va = DataLoader(ReportDS(va, tok), batch_size=BS * 2, num_workers=2)

    model = ReportTeacher().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(dl_tr) * EPOCHS
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    lossf = nn.BCEWithLogitsLoss()

    best, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        for b in dl_tr:
            b = {k: v.to(DEV, non_blocking=True) for k, v in b.items()}
            with torch.cuda.amp.autocast():
                loss = lossf(model(b["input_ids"], b["attention_mask"], b["rule"]),
                             b["target"])
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()

        model.eval(); P = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for b in dl_va:
                b = {k: v.to(DEV) for k, v in b.items()}
                P.append(torch.sigmoid(model(b["input_ids"], b["attention_mask"],
                                             b["rule"])).float().cpu().numpy())
        P = np.concatenate(P)
        m, per = kc.macro_auc(va[kc.LABELS].values.astype(np.float32), P)
        print(f"  fold {fold} ep {ep}: macro AUC {m:.5f}")
        if m > best:
            best, best_state = m, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pred = P
    torch.save(best_state, f"{OUT}/teacher_f{fold}.pt")
    del model; gc.collect(); torch.cuda.empty_cache()
    return best, best_pred, va["StudyInstanceUID"].values

# --- CELL 3 -----------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
from knee_data import make_folds
folds = make_folds(lab_df, FOLDS, seed=42)

oof = np.zeros((len(lab_df), kc.N_LABELS), np.float32)
uid2row = {u: i for i, u in enumerate(lab_df["StudyInstanceUID"])}
scores = []
for f in range(FOLDS):
    s, p, uids = run_fold(f, folds, tok)
    scores.append(s)
    for u, row in zip(uids, p):
        oof[uid2row[u]] = row

m, per = kc.macro_auc(lab_df[kc.LABELS].values.astype(np.float32), oof)
print("\nTEACHER out-of-fold (step 1 rules were %.4f):" % kc.macro_auc(
    lab_df[kc.LABELS].values.astype(np.float32), lab_df[rule_cols].values)[0])
kc.print_auc_table(m, per)

# --- CELL 4 -----------------------------------------------------------------
# Soft-label every study that has no gold label, averaging the five folds.
unl = train[~is_lab].reset_index(drop=True)
dl = DataLoader(ReportDS(unl, tok, with_target=False), batch_size=BS * 2, num_workers=2)
acc = np.zeros((len(unl), kc.N_LABELS), np.float32)
for f in range(FOLDS):
    model = ReportTeacher().to(DEV)
    model.load_state_dict(torch.load(f"{OUT}/teacher_f{f}.pt", map_location="cpu")); model.eval()
    P = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for b in dl:
            b = {k: v.to(DEV) for k, v in b.items()}
            P.append(torch.sigmoid(model(b["input_ids"], b["attention_mask"],
                                         b["rule"])).float().cpu().numpy())
    acc += np.concatenate(P) / FOLDS
    del model; gc.collect(); torch.cuda.empty_cache()

soft = pd.DataFrame(acc, columns=kc.LABELS)
soft.insert(0, "StudyInstanceUID", unl["StudyInstanceUID"].values)
soft["is_gold"] = 0
gold = lab_df[["StudyInstanceUID"] + kc.LABELS].copy()
gold["is_gold"] = 1
targets = pd.concat([gold, soft], ignore_index=True)
targets.to_parquet(f"{OUT}/study_targets.parquet", index=False)

print("\nstudy_targets.parquet", targets.shape)
print("mean soft prevalence vs gold prevalence")
for c in kc.LABELS:
    print(f"  {c:<18s} soft {soft[c].mean():.3f}   gold {gold[c].mean():.3f}")
