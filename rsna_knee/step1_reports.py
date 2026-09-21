# =============================================================================
# STEP 1 - mine the radiology reports.             Kaggle notebook, CPU is fine.
# Internet: ON.  Runtime: ~10 minutes.
#
# Purpose: labels exist for only a small slice of train.csv, but a report exists
# for every study.  Before throwing a transformer at the problem, measure how
# far a precise multilingual rule lexicon gets you.  That number is the floor
# the text teacher in step 2 has to beat, and the rule hits become extra input
# features for it.
#
# The lexicon itself lives in knee_text.py so it can be unit-tested - run
# `python selftest.py` after editing it.
# =============================================================================

# --- CELL 1 -----------------------------------------------------------------
# Bootstrap.  Uses the step-00 output when it is attached, and fetches the
# modules itself when it is not - so this notebook runs standalone with internet
# ON, and off the attached output when internet is OFF.
import os, sys, glob, subprocess

REPO = "https://github.com/rahpalrah/ResumeParser"
BRANCH = "claude/knee-mri-abnormalities-kaggle-6jzvyk"

# An ATTACHED dataset wins, because step 5 runs with no network. Otherwise
# clone fresh on every run - never reuse an earlier run's copy in
# /kaggle/working, or re-running this cell after a module was fixed upstream
# would silently keep the stale code.
def _attached_code():
    hits = glob.glob("/kaggle/input/*/knee_common.py")
    return os.path.dirname(hits[0]) if hits else None

CODE_DIR = _attached_code()
if CODE_DIR is None:
    print("no attached code dataset - cloning fresh (needs internet ON)")
    subprocess.run(["rm", "-rf", "/kaggle/tmp/repo"], check=False)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO,
                    "/kaggle/tmp/repo"], check=True)
    subprocess.run("cp /kaggle/tmp/repo/rsna_knee/*.py /kaggle/working/",
                   shell=True, check=True)
    CODE_DIR = "/kaggle/working"
assert os.path.exists(os.path.join(CODE_DIR, "knee_common.py")), (
    "Could not obtain the modules. Either turn internet ON, or run step 00 and "
    "attach its output (Add Data -> Your Work -> Notebook Output).")

# Drop any already-imported copy: Python caches modules, so without this a
# re-run of this cell keeps the version imported earlier in the session.
for _m in [m for m in list(sys.modules) if m.startswith("knee_")]:
    del sys.modules[_m]
sys.path.insert(0, CODE_DIR)
print("code from:", CODE_DIR)

import re
import numpy as np, pandas as pd
import knee_common as kc
from knee_text import ANATOMY, norm_text, rule_features

COMP = kc.find_comp_dir()          # autodetected: never hard-code the slug
print("competition data:", COMP)
OUT = "/kaggle/working"
train = pd.read_csv(f"{COMP}/train.csv")
train["rep"] = train["Report"].map(norm_text)

print("report length (chars):", train["rep"].str.len().describe().to_string())
print("\nlanguage mix (crude stopword probe):")
for w in ["the ", "der ", " le ", " el ", " de ", " il ", " ve ", " и ", "的"]:
    print(f"  {w!r:8s} {train['rep'].str.contains(w, regex=False).mean():6.1%}")

print("\nanatomy-term coverage (share of reports mentioning the structure at all):")
for lab, pat in ANATOMY.items():
    print(f"  {lab:<18s} {train['rep'].str.contains(pat, regex=True).mean():6.1%}")

# --- CELL 2 -----------------------------------------------------------------
from tqdm.auto import tqdm
tqdm.pandas()
R = np.stack(train["rep"].progress_map(rule_features).values)

rule_df = pd.DataFrame(R, columns=[f"rule_{c}" for c in kc.LABELS])
rule_df.insert(0, "StudyInstanceUID", train["StudyInstanceUID"].values)
rule_df.to_parquet(f"{OUT}/rule_features.parquet", index=False)
print("saved -> rule_features.parquet")

# Gold labels are sparse and may be annotated per column on different subsets,
# so score per cell with a mask rather than restricting to complete rows.
GOLD_M = train[kc.LABELS].notna().values.astype(np.float32)
Y = train[kc.LABELS].fillna(0.0).values.astype(np.float32)

print(f"\nstudies with >=1 gold label : {int((GOLD_M.max(axis=1) > 0).sum()):,}")
print(f"studies with all twelve      : "
      f"{int((GOLD_M.sum(axis=1) == kc.N_LABELS).sum()):,}")
print("\nannotated cells per label:")
for i, c in enumerate(kc.LABELS):
    n = int(GOLD_M[:, i].sum())
    pos = int(Y[GOLD_M[:, i] > 0, i].sum())
    print(f"  {c:<18s} {n:>5} annotated, {pos:>4} positive")

macro, per = kc.macro_auc(Y, R, mask=GOLD_M)
print(f"\nRULE-ONLY baseline (the floor the teacher must beat):")
kc.print_auc_table(macro, per)

# --- CELL 3 -----------------------------------------------------------------
# Precision matters more than recall: the rules are both the teacher's training
# target and an input feature, and a noisy target is worse than a sparse one.
from sklearn.metrics import precision_score, recall_score
print("per-label precision / recall, over annotated cells only")
prec = {}
for i, c in enumerate(kc.LABELS):
    sel = GOLD_M[:, i] > 0
    if sel.sum() == 0:
        print(f"  {c:<18s} no annotated cells")
        continue
    p = precision_score(Y[sel, i], R[sel, i], zero_division=0)
    r = recall_score(Y[sel, i], R[sel, i], zero_division=0)
    prec[c] = p
    print(f"  {c:<18s} P={p:.3f} R={r:.3f}  fires {R[sel, i].mean():.1%} "
          f"of annotated, {R[:, i].mean():.1%} of all 4,407  true {Y[sel, i].mean():.1%}")

# A label below ~0.6 precision is a lexicon bug, not a hard label.  These are
# the reports to read before editing knee_text.py.
if prec:
    worst = min(prec, key=prec.get)
    wi = kc.LABELS.index(worst)
    sel = GOLD_M[:, wi] > 0
    print(f"\nworst precision: {worst} ({prec[worst]:.3f}) - false positives:")
    fp = np.where(sel & (R[:, wi] == 1) & (Y[:, wi] == 0))[0][:3]
    for j in fp:
        print("   ...", train.iloc[j]["rep"][:300], "\n")
    print(f"missed positives ({worst}):")
    fn = np.where(sel & (R[:, wi] == 0) & (Y[:, wi] == 1))[0][:3]
    for j in fn:
        print("   ...", train.iloc[j]["rep"][:300], "\n")
