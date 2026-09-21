# =============================================================================
# ONE-CELL RUNNER. Paste THIS into Kaggle, not the step itself.
#
# The steps are long enough that a copy-paste can be truncated - a cell ending
# mid-docstring fails with "SyntaxError: incomplete input", which looks like a
# bug in the code and is not. This clones the repo and runs the step from the
# file, so what executes is always the whole, current version.
#
# To override a setting, assign it BEFORE the exec: the steps read their
# settings from globals() first.  e.g.  SHARD = 1
# =============================================================================
import os, sys, subprocess

STEP = "step3_preprocess.py"     # <-- the only thing to change between steps
REPO = "https://github.com/rahpalrah/ResumeParser"
BRANCH = "claude/knee-mri-abnormalities-kaggle-6jzvyk"

subprocess.run(["rm", "-rf", "/kaggle/tmp/repo"], check=False)
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO,
                "/kaggle/tmp/repo"], check=True, timeout=180)
subprocess.run("cp /kaggle/tmp/repo/rsna_knee/*.py /kaggle/working/",
               shell=True, check=True)
for _m in [m for m in list(sys.modules) if m.startswith("knee_")]:
    del sys.modules[_m]
sys.path.insert(0, "/kaggle/working")

_src = open(f"/kaggle/working/{STEP}", encoding="utf-8").read()
print(f"running {STEP} ({len(_src.splitlines())} lines)\n" + "=" * 66)
exec(compile(_src, STEP, "exec"))
