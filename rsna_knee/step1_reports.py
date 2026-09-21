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

lab = train[kc.LABELS].notna().all(axis=1).values
y = train.loc[lab, kc.LABELS].values.astype(np.float32)
macro, per = kc.macro_auc(y, R[lab])
print(f"\nRULE-ONLY baseline on {int(lab.sum()):,} labelled studies (this is the floor):")
kc.print_auc_table(macro, per)

# --- CELL 3 -----------------------------------------------------------------
# Precision matters more than recall here: the rules are a feature for the
# transformer, and a noisy feature is worse than a sparse one.
from sklearn.metrics import precision_score, recall_score
print("per-label precision / recall / positive rate of the rules")
for i, c in enumerate(kc.LABELS):
    p = precision_score(y[:, i], R[lab][:, i], zero_division=0)
    r = recall_score(y[:, i], R[lab][:, i], zero_division=0)
    print(f"  {c:<18s} P={p:.3f} R={r:.3f}  fires {R[lab][:, i].mean():.1%}  "
          f"true {y[:, i].mean():.1%}")

# Any label where precision is below ~0.6 is a lexicon bug, not a hard label.
# Print a few misfires and extend ANATOMY / ABNORMAL in knee_text.py.
worst = min(range(kc.N_LABELS),
            key=lambda i: precision_score(y[:, i], R[lab][:, i], zero_division=1))
print(f"\nworst-precision label: {kc.LABELS[worst]} - three false positives:")
fp = np.where((R[lab][:, worst] == 1) & (y[:, worst] == 0))[0][:3]
for j in fp:
    print("   ...", train.loc[lab].iloc[j]["rep"][:300], "\n")

print("saved -> rule_features.parquet")
