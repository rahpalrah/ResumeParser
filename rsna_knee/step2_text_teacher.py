# =============================================================================
# STEP 2 - the report teacher.                     Kaggle notebook, GPU T4 x2.
# Internet: ON (downloads xlm-roberta-base).  Runtime: ~1 h.
#
# WHY THIS IS NOT ORDINARY SUPERVISED FINE-TUNING
#
# train.csv carries 4,407 reports and gold labels on only ~58 studies. Fitting
# a 278M-parameter encoder to 58 examples across 12 targets memorises them; it
# does not learn to read a report. So the teacher is trained by SELF-TRAINING
# instead:
#
#   1. The step-1 rule lexicon labels all 4,407 reports. Rules are precise and
#      low-recall - they fire on the phrasings somebody thought of.
#   2. XLM-R is fitted to those 4,407 RULE labels. With thousands of noisy
#      examples it generalises past the lexicon: it learns the paraphrases,
#      the languages and the hedged phrasings the regexes miss, because those
#      co-occur with the ones that do fire.
#   3. The gold studies are held out of every fit and used ONLY to measure. At
#      n=58 they are a sanity check, not a selection signal - do not tune
#      against them.
#
# Output: study_targets.parquet, carrying for every study a soft target per
# label plus a per-cell is_gold flag, so step 4 can weight a real annotation
# above a teacher guess.
# =============================================================================

# --- CELL 1 -----------------------------------------------------------------
# !pip install -q iterative-stratification
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
                  "step 00 and attach its output.")
sys.path.insert(0, CODE_DIR)
print("code from:", CODE_DIR)

import gc, math, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
import knee_common as kc
from knee_text import norm_text, rule_features

COMP = kc.find_comp_dir()
print("competition data:", COMP)
OUT = "/kaggle/working"

MODEL_NAME = "xlm-roberta-base"   # the reports are in several languages
MAX_LEN = 512
EPOCHS = 3
BS = 8
LR = 2e-5
VAL_FRAC = 0.1
DEV = "cuda"
kc.seed_everything(42)

train = pd.read_csv(f"{COMP}/train.csv")
_r = [p for p in glob.glob("/kaggle/input/*/rule_features.parquet")
      + [f"{OUT}/rule_features.parquet"] if os.path.exists(p)]
assert _r, "Attach the step-1 notebook output (rule_features.parquet)"
rules = pd.read_parquet(_r[0])
train = train.merge(rules, on="StudyInstanceUID", how="left")
rule_cols = [f"rule_{c}" for c in kc.LABELS]
train[rule_cols] = train[rule_cols].fillna(0.0)

# --- CELL 2 -----------------------------------------------------------------
# Gold cells are sparse: a study may be annotated for some findings and not
# others.  Track them per cell, never per study.
GOLD = train[kc.LABELS].notna().values.astype(np.float32)     # (N, 12)
gold_rows = GOLD.max(axis=1) > 0
print(f"studies with at least one gold label : {int(gold_rows.sum()):,}")
print(f"studies with all twelve              : "
      f"{int((GOLD.sum(axis=1) == kc.N_LABELS).sum()):,}")
print("\ngold annotations per label:")
for i, c in enumerate(kc.LABELS):
    n = int(GOLD[:, i].sum())
    pos = float(np.nansum(train[c].values[GOLD[:, i] > 0]))
    print(f"  {c:<18s} {n:>5} annotated, {pos:>4.0f} positive")

# The teacher's training targets are the RULES, on the studies with no gold.
Y_RULE = train[rule_cols].values.astype(np.float32)
fit_df = train[~gold_rows].reset_index(drop=True)
gold_df = train[gold_rows].reset_index(drop=True)
print(f"\nfitting on {len(fit_df):,} rule-labelled studies; "
      f"{len(gold_df):,} gold studies held out entirely")

rng = np.random.default_rng(0)
perm = rng.permutation(len(fit_df))
n_val = int(len(fit_df) * VAL_FRAC)
va_idx, tr_idx = perm[:n_val], perm[n_val:]
print(f"early-stopping split: {len(tr_idx):,} train / {len(va_idx):,} val")

# --- CELL 3 -----------------------------------------------------------------
class ReportDS(Dataset):
    def __init__(self, df, tok, targets=None):
        self.txt = df["Report"].fillna("").astype(str).tolist()
        self.rule = df[rule_cols].values.astype(np.float32)
        self.y = targets
        self.tok = tok

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, i):
        enc = self.tok(self.txt[i], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        item = {k: v[0] for k, v in enc.items()}
        item["rule"] = torch.from_numpy(self.rule[i])
        if self.y is not None:
            item["target"] = torch.from_numpy(self.y[i])
        return item


class ReportTeacher(nn.Module):
    """Encoder mean-pool concatenated with the rule hits.

    Feeding the rules in explicitly is not redundant with the encoder: it gives
    the head a calibrated, language-agnostic prior, and it means the teacher can
    never do worse than the rules on a phrasing the rules already handle."""

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


tok = AutoTokenizer.from_pretrained(MODEL_NAME)
y_fit = fit_df[rule_cols].values.astype(np.float32)
dl_tr = DataLoader(ReportDS(fit_df.iloc[tr_idx], tok, y_fit[tr_idx]),
                   batch_size=BS, shuffle=True, num_workers=2,
                   drop_last=True, pin_memory=True)
dl_va = DataLoader(ReportDS(fit_df.iloc[va_idx], tok, y_fit[va_idx]),
                   batch_size=BS * 2, num_workers=2)
dl_gold = DataLoader(ReportDS(gold_df, tok), batch_size=BS * 2, num_workers=2)

GOLD_Y = gold_df[kc.LABELS].fillna(0.0).values.astype(np.float32)
GOLD_M = gold_df[kc.LABELS].notna().values.astype(np.float32)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    P = []
    with kc.amp_autocast():
        for b in loader:
            b = {k: v.to(DEV) for k, v in b.items() if k != "target"}
            P.append(torch.sigmoid(model(b["input_ids"], b["attention_mask"],
                                         b["rule"])).float().cpu().numpy())
    return np.concatenate(P)


model = ReportTeacher().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = len(dl_tr) * EPOCHS
sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = kc.make_grad_scaler()
lossf = nn.BCEWithLogitsLoss()

rule_gold_auc, _ = kc.macro_auc(GOLD_Y, gold_df[rule_cols].values.astype(np.float32),
                                mask=GOLD_M)
print(f"rules alone, measured on the gold studies: {rule_gold_auc:.5f}\n")

best = -1.0
for ep in range(EPOCHS):
    model.train(); t0 = time.time(); run = 0.0
    for i, b in enumerate(dl_tr):
        b = {k: v.to(DEV, non_blocking=True) for k, v in b.items()}
        with kc.amp_autocast():
            loss = lossf(model(b["input_ids"], b["attention_mask"], b["rule"]),
                         b["target"])
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
        run += loss.item()
        if (i + 1) % 100 == 0:
            print(f"    ep{ep} {i+1}/{len(dl_tr)} loss {run/(i+1):.4f} "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    P_va = predict(model, dl_va)
    m_rule, _ = kc.macro_auc(y_fit[va_idx], P_va)
    P_gold = predict(model, dl_gold)
    m_gold, per_gold = kc.macro_auc(GOLD_Y, P_gold, mask=GOLD_M)
    print(f"  epoch {ep}: held-out rule AUC {m_rule:.5f} | GOLD AUC {m_gold:.5f} "
          f"| {(time.time()-t0)/60:.1f} min")
    if m_rule > best:                      # selection on the large noisy set,
        best = m_rule                      # never on 58 gold studies
        torch.save(model.state_dict(), f"{OUT}/teacher.pt")
        print(f"    saved (held-out rule AUC {best:.5f})")

# --- CELL 4 -----------------------------------------------------------------
model.load_state_dict(torch.load(f"{OUT}/teacher.pt", map_location="cpu"))
model = model.to(DEV)

P_gold = predict(model, dl_gold)
m_gold, per_gold = kc.macro_auc(GOLD_Y, P_gold, mask=GOLD_M)
print(f"TEACHER on the {len(gold_df)} gold studies "
      f"(rules alone were {rule_gold_auc:.4f}):")
kc.print_auc_table(m_gold, per_gold)
# A continuation line must never begin with % or ! - IPython reads those as
# cell magics and the cell dies with a SyntaxError.
print(f"\n  n={len(gold_df)}: a sanity check, not a leaderboard estimate.")

dl_all = DataLoader(ReportDS(train, tok), batch_size=BS * 2, num_workers=2)
SOFT = predict(model, dl_all)

# Gold wins wherever it exists; the teacher fills every other cell.
GOLD_VALS = train[kc.LABELS].fillna(0.0).values.astype(np.float32)
TARGET = np.where(GOLD > 0, GOLD_VALS, SOFT)

targets = pd.DataFrame(TARGET, columns=kc.LABELS)
targets.insert(0, "StudyInstanceUID", train["StudyInstanceUID"].values)
for i, c in enumerate(kc.LABELS):
    targets[f"is_gold_{c}"] = GOLD[:, i]
targets.to_parquet(f"{OUT}/study_targets.parquet", index=False)

print(f"\nstudy_targets.parquet {targets.shape}")
print(f"  gold cells    : {int(GOLD.sum()):,} of {GOLD.size:,}")
print("\nteacher mean probability vs rule firing rate, per label")
for i, c in enumerate(kc.LABELS):
    print(f"  {c:<18s} teacher {SOFT[:, i].mean():.3f}   "
          f"rules {Y_RULE[:, i].mean():.3f}")
