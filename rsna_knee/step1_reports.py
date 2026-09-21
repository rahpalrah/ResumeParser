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
# The rules are graded now (0 / 0.35 / 0.7 / 1.0 by severity), so precision and
# recall need a hard threshold. The AUC above uses the grades directly.
RB = (R > 0.5).astype(np.float32)
print("per-label precision / recall at threshold 0.5, over annotated cells only")
prec = {}
for i, c in enumerate(kc.LABELS):
    sel = GOLD_M[:, i] > 0
    if sel.sum() == 0:
        print(f"  {c:<18s} no annotated cells")
        continue
    p = precision_score(Y[sel, i], RB[sel, i], zero_division=0)
    r = recall_score(Y[sel, i], RB[sel, i], zero_division=0)
    prec[c] = p
    print(f"  {c:<18s} P={p:.3f} R={r:.3f}  fires {RB[sel, i].mean():.1%} "
          f"of annotated, {RB[:, i].mean():.1%} of all 4,407  true {Y[sel, i].mean():.1%}")

# A label below ~0.6 precision is a lexicon bug, not a hard label.  These are
# the reports to read before editing knee_text.py.
if prec:
    worst = min(prec, key=prec.get)
    wi = kc.LABELS.index(worst)
    sel = GOLD_M[:, wi] > 0
    print(f"\nworst precision: {worst} ({prec[worst]:.3f}) - false positives:")
    fp = np.where(sel & (RB[:, wi] == 1) & (Y[:, wi] == 0))[0][:3]
    for j in fp:
        print("   ...", train.iloc[j]["rep"][:300], "\n")
    print(f"missed positives ({worst}):")
    fn = np.where(sel & (RB[:, wi] == 0) & (Y[:, wi] == 1))[0][:3]
    for j in fn:
        print("   ...", train.iloc[j]["rep"][:300], "\n")

# --- CELL 4 -----------------------------------------------------------------
# Where the lexicon still misses, by script.  Aggregate coverage hides which
# language a gap belongs to, and a gap in a language is a terminology list
# somebody has to write - so measure it directly and print the reports that
# fall through.
# Named by what the corpus actually holds: the Cyrillic block is Bulgarian
# (медиален, no soft sign), and Greek and Croatian hide inside Latin script.
CYR = train["rep"].str.contains(r"[а-я]", regex=True)
GRK = train["rep"].str.contains(r"[α-ω]", regex=True) & ~CYR
TRK = train["rep"].str.contains(r"\b(?:ve|ile|mevcut|izlen\w+|saptan\w+|eklem)\b",
                                regex=True) & ~CYR & ~GRK
HRV = train["rep"].str.contains(r"\b(?:uredan|urednog|bez znakova|menisk\b|krizni)\b",
                                regex=True) & ~CYR & ~GRK & ~TRK
OTH = ~CYR & ~GRK & ~TRK & ~HRV
for name, sel in [("bulgarian (cyrillic)", CYR), ("greek", GRK), ("turkish", TRK),
                  ("croatian-ish", HRV), ("other latin", OTH)]:
    n = int(sel.sum())
    print(f"\n{name}: {n:,} reports ({n/len(train):.1%})")
    if n == 0:
        continue
    for lab, pat in ANATOMY.items():
        cov = train.loc[sel, "rep"].str.contains(pat, regex=True).mean()
        print(f"   {lab:<18s} {cov:6.1%}")

# --- CELL 5 -----------------------------------------------------------------
# Read the reports the lexicon cannot see.  Guessing medical vocabulary from
# memory is how a pattern like \bmm\b gets written; this prints the actual
# sentences so the terminology list is copied, not invented.
BUCKET = "other"        # cyrillic | greek | turkish | croatian | other | all
PROBE = "ACL"           # structure that must be missing; None = any report
N_SHOW = 6
CHARS = 500

sel = {"cyrillic": CYR, "greek": GRK, "turkish": TRK, "croatian": HRV,
       "other": OTH, "all": pd.Series(True, index=train.index)}[BUCKET]
if PROBE:
    sel = sel & ~train["rep"].str.contains(ANATOMY[PROBE], regex=True)

sub = train[sel]
print(f"{BUCKET}: {len(sub):,} reports"
      + (f" with no {PROBE} match" if PROBE else ""))
if len(sub):
    for t in sub["rep"].sample(min(N_SHOW, len(sub)), random_state=0):
        print("\n" + "-" * 72)
        print(t[:CHARS].strip())
