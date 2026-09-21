# =============================================================================
# STEP 00 - bootstrap.                             Kaggle notebook, CPU, ~3 min.
# Internet: ON.  Run this FIRST, and again any time you edit a knee_*.py.
#
# Produces one artifact that every later notebook consumes: a folder holding
# the four library modules plus the pip wheels the offline submission notebook
# needs.  Nothing here requires a machine of your own.
#
# When it finishes: Save Version (Save & Run All).  Later notebooks attach this
# notebook's output via  Add Data -> Your Work -> Notebook Output.  They locate
# it by globbing /kaggle/input/*/knee_common.py, so it does not matter what
# Kaggle ends up calling it.
# =============================================================================

# --- CELL 1 -----------------------------------------------------------------
import os, shutil, subprocess, sys, glob

REPO = "https://github.com/rahpalrah/ResumeParser"
BRANCH = "claude/knee-mri-abnormalities-kaggle-6jzvyk"
WORK = "/kaggle/working"

subprocess.run(["rm", "-rf", "/kaggle/tmp/repo"], check=False)
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO,
                "/kaggle/tmp/repo"], check=True)

for src in sorted(glob.glob("/kaggle/tmp/repo/rsna_knee/*.py")) + \
           sorted(glob.glob("/kaggle/tmp/repo/rsna_knee/*.md")):
    shutil.copy(src, WORK)
    print("copied", os.path.basename(src))
sys.path.insert(0, WORK)

# --- CELL 2 -----------------------------------------------------------------
# The self-test. 15 checks on synthetic data - no competition data, no GPU.
# If this does not end in ALL SELF-TESTS PASSED, stop here: something in the
# modules is broken and every later step would inherit it.
!cd /kaggle/working && python selftest.py

# --- CELL 3 -----------------------------------------------------------------
# Wheels for the DICOM decoders.  Step 5 runs with internet OFF and still has to
# decode JPEG Lossless and JPEG 2000, so the wheels have to travel with the code.
!mkdir -p /kaggle/working/wheels
!pip download -q -d /kaggle/working/wheels \
    pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm pydicom

import os
w = sorted(os.listdir("/kaggle/working/wheels"))
print(f"{len(w)} wheels, {sum(os.path.getsize('/kaggle/working/wheels/'+f) for f in w)/2**20:.1f} MiB")
for f in w:
    print("  ", f)

# Prove they install with no network, exactly as step 5 will.
!pip install -q --no-index --find-links=/kaggle/working/wheels \
    pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm

import knee_common as kc
print("\noffline decoder check:")
kc.check_dicom_backends()

# --- CELL 4 -----------------------------------------------------------------
print("""
BOOTSTRAP COMPLETE.

  1. Save Version  ->  Save & Run All (Commit).
  2. In every later notebook:  Add Data -> Your Work -> Notebook Output ->
     this notebook.
  3. Re-run this notebook whenever you edit a knee_*.py in the repo, then
     re-attach the newer version.

Contents of /kaggle/working that later steps use:""")
for f in sorted(os.listdir("/kaggle/working")):
    print("   ", f)
