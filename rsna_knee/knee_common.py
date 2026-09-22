"""
knee_common.py - shared library for the RSNA Knee Abnormalities Detection pipeline.

Design notes
------------
This module is written so it can be dropped into a Kaggle notebook as a single
cell (`%%writefile knee_common.py`) and imported by every step of the pipeline,
including the offline submission notebook.  It has no dependency beyond what a
stock Kaggle image already ships: numpy, pandas, opencv, torch, timm, pydicom.

Coordinate / laterality convention
----------------------------------
Images are padded to square and resized, never centre-cropped, so the medial /
lateral halves of a coronal or axial slice always map to the left / right half
of the tensor.  A horizontal flip therefore swaps medial and lateral, which is
why `flip_lr` in the dataset also flips the laterality token fed to the model.
"""
from __future__ import annotations

import glob
import math
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LABELS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
N_LABELS = len(LABELS)

# Labels whose name refers to one tibiofemoral compartment.  These are the ones
# the compartment-aware branch of the model is asked to explain.
MEDIAL_LABELS = ["MCL", "Medial Meniscus", "Medial OA"]
LATERAL_LABELS = ["Lateral Meniscus", "Lateral OA"]

PLANES = ["Sagittal", "Coronal", "Axial"]
PLANE2IDX = {p: i for i, p in enumerate(PLANES)}

# One "slot" = one (plane, fluid_sensitive) role the study model always expects.
# Keeping the slot order fixed makes the series tokens positionally meaningful
# and lets a study with missing sequences be handled by masking rather than by
# a different code path.
SLOTS: List[Tuple[str, int]] = [
    ("Sagittal", 1),   # sag fluid-sensitive: menisci, ACL, effusion, contusion
    ("Coronal", 1),    # cor fluid-sensitive: MCL, compartment OA, contusion
    ("Axial", 1),      # ax fluid-sensitive: PF OA, synovitis, Baker's
    ("Sagittal", 0),   # sag T1/PD: cartilage, marrow, fracture line
]
N_SLOTS = len(SLOTS)


@dataclass
class Cfg:
    """Everything that changes between the accuracy run and the efficiency run."""
    # preprocessing
    img_size: int = 256          # sprite tile size (square)
    n_slices: int = 16           # slices kept per series
    grid_w: int = 4              # sprite columns; rows = ceil(n_slices / grid_w)
    jpeg_quality: int = 92

    # model
    backbone: str = "convnext_tiny.fb_in22k_ft_in1k"
    pretrained: bool = True
    embed_dim: int = 256
    slice_layers: int = 2
    study_layers: int = 2
    n_heads: int = 8
    drop: float = 0.1
    drop_path: float = 0.1
    use_compartment: bool = True
    use_coupling: bool = True

    # training
    folds: int = 5
    epochs_pseudo: int = 2
    epochs_gold: int = 6
    batch_size: int = 2
    accum: int = 8
    lr_backbone: float = 1e-4
    lr_head: float = 5e-4
    weight_decay: float = 1e-2
    pseudo_weight: float = 0.5
    ema_decay: float = 0.999
    seed: int = 42
    num_workers: int = 2

    # paths (overridden per notebook)
    comp_dir: str = ""           # set by find_comp_dir(); never hard-coded
    cache_dir: str = "/kaggle/working/cache"
    out_dir: str = "/kaggle/working"

    @property
    def grid_h(self) -> int:
        return math.ceil(self.n_slices / self.grid_w)


CFG_EFFICIENCY = Cfg(
    img_size=192, n_slices=10, grid_w=5,
    backbone="efficientnetv2_rw_t.ra2_in1k",
    embed_dim=192, slice_layers=1, study_layers=1,
    batch_size=4, accum=4,
)


def amp_autocast(device: str = "cuda", enabled: bool = True):
    """Mixed-precision context that works across torch versions.

    `torch.cuda.amp.autocast` was deprecated in torch 2.4 in favour of
    `torch.amp.autocast("cuda")` and is on its way out.  Kaggle's image moves
    faster than this code does - it is already on torch 2.10 - so ask for the
    modern spelling and fall back to the old one.
    """
    import torch
    try:
        return torch.amp.autocast(device, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(device: str = "cuda"):
    """GradScaler, same version dance as amp_autocast."""
    import torch
    try:
        return torch.amp.GradScaler(device)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def find_inputs(name: str, root: str = "/kaggle/input", max_depth: int = 8) -> List[str]:
    """Every attached file or directory matching `name`, at any depth.

    Kaggle mounts things at depths that vary by kind: a competition at
    /kaggle/input/competitions/<slug>/, a dataset at /kaggle/input/<slug>/ or
    /kaggle/input/datasets/<owner>/<slug>/, and a NOTEBOOK OUTPUT under
    /kaggle/input/notebooks/<user>/<notebook>/... deeper still. Globbing a
    fixed set of depths finds nothing the moment the layout is one level past
    whatever was guessed, so walk instead.

    The walk prunes the DICOM trees. train_series holds ~820,000 files across
    tens of thousands of directories, and descending into it to look for a
    parquet would take minutes and find nothing.
    """
    import fnmatch

    skip = {"train_series", "test_series", ".git", "__pycache__"}
    out: List[str] = []
    frontier = [(root, 0)]
    while frontier:
        d, depth = frontier.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            matched = fnmatch.fnmatch(e.name, name)
            if matched:
                out.append(e.path)
            # Do not descend into a directory that already matched, nor into
            # the DICOM trees, nor past the depth cap.
            if (e.is_dir() and not matched and depth + 1 <= max_depth
                    and e.name not in skip):
                frontier.append((e.path, depth + 1))
    return sorted(set(out))


def find_comp_dir(root: str = "/kaggle/input", marker: str = "sample_submission.csv",
                  verbose: bool = True) -> str:
    """Locate the competition data wherever Kaggle chose to mount it.

    Kaggle has used several layouts (/kaggle/input/<slug>/ and
    /kaggle/input/competitions/<slug>/), and the slug itself is easy to mistype
    - "abnormality" vs "abnormalities" costs a debugging cycle every time. So
    nothing in this pipeline hard-codes the path.

    Candidates are SCORED rather than taken first-match, because a derived
    dataset (someone's preprocessed JPEG export of this same competition)
    legitimately carries copies of train.csv, train_series.csv and
    sample_submission.csv. Only the real competition mount also carries the
    DICOM directories, so that is what decides it.

    Set KNEE_COMP_DIR to override entirely.
    """
    override = os.environ.get("KNEE_COMP_DIR")
    if override:
        if not os.path.isdir(override):
            raise FileNotFoundError(f"KNEE_COMP_DIR={override} is not a directory")
        return override

    hits = find_inputs(marker, root)
    if not hits:
        raise FileNotFoundError(
            f"No {marker} found under {root}. Add the competition data to this "
            "notebook: Add Data -> Competitions -> RSNA Knee Abnormalities Detection."
        )

    def score(d: str) -> int:
        pts = 0
        if os.sep + "competitions" + os.sep in d + os.sep:
            pts += 4
        if os.path.isdir(os.path.join(d, "train_series")) or \
           os.path.isdir(os.path.join(d, "test_series")):
            pts += 3
        if os.path.exists(os.path.join(d, "train_series.csv")):
            pts += 2
        if os.path.exists(os.path.join(d, "train.csv")):
            pts += 1
        return pts

    cands = sorted({os.path.dirname(h) for h in hits},
                   key=lambda d: (-score(d), d))
    if verbose and len(cands) > 1:
        print("several candidate data directories; scored:")
        for d in cands:
            print(f"   {score(d):>2}  {d}")
    return cands[0]


def find_series_root(comp_dir: str, split: str) -> str:
    """Directory holding the per-study DICOM folders for `split`."""
    for name in (f"{split}_series", f"{split}_images", split):
        p = os.path.join(comp_dir, name)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"no {split}_series/ directory under {comp_dir}")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# DICOM I/O
# --------------------------------------------------------------------------- #

def check_dicom_backends(verbose: bool = True) -> Dict[str, bool]:
    """Report which pixel-data decoders are importable.

    The competition mixes Explicit/Implicit VR Little Endian (needs nothing),
    JPEG Lossless (needs pylibjpeg-libjpeg or gdcm) and JPEG 2000 (needs
    pylibjpeg-openjpeg or gdcm).  Missing one of these does not raise at import
    time - it raises when you touch `.pixel_array` on an affected series, which
    is why this check exists as an explicit step.
    """
    status = {}
    for mod in ["pydicom", "pylibjpeg", "libjpeg", "openjpeg", "gdcm"]:
        try:
            __import__(mod)
            status[mod] = True
        except Exception:
            status[mod] = False
    if verbose:
        for k, v in status.items():
            print(f"  {'OK  ' if v else 'MISS'} {k}")
    return status


def _slice_position(ds) -> float:
    """Position of a slice along the acquisition normal, for reliable ordering."""
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is not None and ipp is not None and len(iop) == 6 and len(ipp) == 3:
        r = np.asarray(iop[:3], dtype=np.float64)
        c = np.asarray(iop[3:], dtype=np.float64)
        n = np.cross(r, c)
        return float(np.dot(np.asarray(ipp, dtype=np.float64), n))
    inst = getattr(ds, "InstanceNumber", None)
    return float(inst) if inst is not None else 0.0


def _pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    top = (s - h) // 2
    left = (s - w) // 2
    out = np.zeros((s, s), dtype=img.dtype)
    out[top:top + h, left:left + w] = img
    return out


def pick_slice_indices(n_avail: int, n_want: int, jitter: bool = False) -> np.ndarray:
    """Uniformly sample `n_want` slice indices spanning the series.

    Series range from ~20 to a few hundred slices; uniform sampling over the
    full extent keeps the anatomical coverage constant regardless of protocol.
    """
    if n_avail <= 0:
        return np.zeros(n_want, dtype=int)
    if n_avail <= n_want:
        idx = np.arange(n_avail)
        # repeat the edges rather than pad with black, which would look like a
        # real (empty) slice to the slice transformer
        pad = n_want - n_avail
        idx = np.concatenate([idx, np.full(pad, n_avail - 1)])
        return idx
    pos = np.linspace(0, n_avail - 1, n_want)
    if jitter:
        step = (n_avail - 1) / max(n_want - 1, 1)
        pos = pos + np.random.uniform(-step / 3, step / 3, size=n_want)
    return np.clip(np.round(pos), 0, n_avail - 1).astype(int)


def load_series_volume(series_dir: str, n_slices: int, img_size: int,
                       jitter: bool = False) -> Optional[np.ndarray]:
    """Read one DICOM series into a (n_slices, img_size, img_size) uint8 volume.

    Headers of every file are read first (cheap, no pixel decode) purely to sort
    the slices; only the `n_slices` selected files are actually decoded.  On a
    300-slice series that is a ~20x saving on decode time.
    """
    import pydicom

    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if not files:
        return None

    heads = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            heads.append((_slice_position(ds), f))
        except Exception:
            continue
    if not heads:
        return None
    heads.sort(key=lambda t: t[0])
    ordered = [f for _, f in heads]

    idx = pick_slice_indices(len(ordered), n_slices, jitter=jitter)
    raw: List[np.ndarray] = []
    invert: List[bool] = []
    n_decoded = 0
    for i in idx:
        try:
            ds = pydicom.dcmread(ordered[int(i)], force=True)
            a = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            a = a * slope + inter
            inv = str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1"
            n_decoded += 1
        except Exception:
            a = raw[-1].copy() if raw else np.zeros((img_size, img_size), np.float32)
            inv = False
        if a.ndim == 3:            # rare multi-frame or RGB slice
            a = a.mean(axis=-1) if a.shape[-1] in (3, 4) else a[a.shape[0] // 2]
        raw.append(a)
        invert.append(inv)

    # A slice that fails to decode is replaced above by its neighbour, or by
    # zeros when it is the first. That is right for one bad slice in a series
    # and catastrophic for a series where NOTHING decodes: without this check
    # the function returns a perfectly black volume and reports success, so a
    # missing transfer-syntax plugin would fill the cache with black sprites,
    # report no failures, pass every gate, and train the model on nothing.
    if n_decoded == 0:
        return None

    # MONOCHROME1 means high value = dark.  Invert against the maximum of the
    # whole stack, never per slice: a per-slice max would rescale every slice
    # differently and destroy the cross-slice intensity relationship that the
    # normalisation below is there to preserve.
    if any(invert):
        vmax = max(float(r.max()) for r in raw)
        raw = [(vmax - r) if inv else r for r, inv in zip(raw, invert)]

    # Robust normalisation over the whole stack, not per slice: relative signal
    # between slices carries information (e.g. a bright effusion vs marrow).
    flat = np.concatenate([r.reshape(-1) for r in raw])
    lo, hi = np.percentile(flat, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(flat.min()), float(flat.max()) + 1e-6

    vol = np.zeros((n_slices, img_size, img_size), dtype=np.uint8)
    for j, a in enumerate(raw):
        a = np.clip((a - lo) / (hi - lo), 0, 1)
        a = _pad_to_square(a)
        a = cv2.resize(a, (img_size, img_size), interpolation=cv2.INTER_AREA)
        vol[j] = (a * 255.0).astype(np.uint8)
    return vol


# --------------------------------------------------------------------------- #
# Sprite cache  (one JPEG per series instead of 30 DICOMs)
# --------------------------------------------------------------------------- #

def vol_to_sprite(vol: np.ndarray, grid_w: int) -> np.ndarray:
    k, h, w = vol.shape
    rows = math.ceil(k / grid_w)
    canvas = np.zeros((rows * h, grid_w * w), dtype=np.uint8)
    for i in range(k):
        r, c = divmod(i, grid_w)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = vol[i]
    return canvas


def sprite_to_vol(sprite: np.ndarray, n_slices: int, img_size: int,
                  grid_w: int) -> np.ndarray:
    vol = np.zeros((n_slices, img_size, img_size), dtype=np.uint8)
    for i in range(n_slices):
        r, c = divmod(i, grid_w)
        vol[i] = sprite[r * img_size:(r + 1) * img_size,
                        c * img_size:(c + 1) * img_size]
    return vol


def write_sprite(path: str, vol: np.ndarray, grid_w: int, quality: int) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sprite = vol_to_sprite(vol, grid_w)
    ok, buf = cv2.imencode(".jpg", sprite, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"failed to encode sprite {path}")
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())
    return int(buf.nbytes)


def read_sprite(path: str, n_slices: int, img_size: int, grid_w: int) -> Optional[np.ndarray]:
    sprite = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if sprite is None:
        return None
    return sprite_to_vol(sprite, n_slices, img_size, grid_w)


def sprite_path(cache_dir: str, study_uid: str, series_uid: str) -> str:
    # shard by the last two characters of the study UID so no directory holds
    # more than a few hundred files (Kaggle datasets dislike flat 27k-file dirs)
    return os.path.join(cache_dir, study_uid[-2:], f"{study_uid}__{series_uid}.jpg")


# --------------------------------------------------------------------------- #
# Series selection
# --------------------------------------------------------------------------- #

def assign_slots(series_df: pd.DataFrame) -> pd.DataFrame:
    """Score every series against each of the fixed (plane, fluid) slots.

    Returns the input frame with an extra `slot` column (-1 = no slot wanted).
    A study rarely has exactly the four sequences we ask for, so the rule is:
    exact (plane, fluid) match first, then same plane with the other contrast,
    and anything left over is dropped.
    """
    df = series_df.copy()
    df["slot"] = -1
    for i, (plane, fluid) in enumerate(SLOTS):
        exact = (df["Anatomical_Plane"] == plane) & (df["Fluid_Sensitive"] == fluid) & (df["slot"] < 0)
        df.loc[exact, "slot"] = i
    for i, (plane, _fluid) in enumerate(SLOTS):
        if (df["slot"] == i).any():
            continue
        same_plane = (df["Anatomical_Plane"] == plane) & (df["slot"] < 0)
        if same_plane.any():
            df.loc[same_plane.idxmax(), "slot"] = i
    return df


def study_series_map(series_df: pd.DataFrame) -> Dict[str, Dict[int, List[str]]]:
    """study_uid -> {slot -> [series_uid, ...]}"""
    out: Dict[str, Dict[int, List[str]]] = {}
    for study, grp in series_df.groupby("StudyInstanceUID"):
        g = assign_slots(grp)
        slots: Dict[int, List[str]] = {}
        for _, row in g[g["slot"] >= 0].iterrows():
            slots.setdefault(int(row["slot"]), []).append(row["SeriesInstanceUID"])
        out[study] = slots
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def macro_auc(y_true: np.ndarray, y_pred: np.ndarray,
              mask: Optional[np.ndarray] = None,
              threshold: float = 0.5) -> Tuple[float, Dict[str, float]]:
    """Competition metric: mean ROC AUC over the twelve targets.

    `mask` (same shape as y_true, 1 = this cell is a real annotation) exists
    because the gold labels are sparse: a study may be annotated for some
    findings and not others, and scoring an un-annotated cell against a
    placeholder would be worse than skipping it.

    `y_true` may be SOFT. The rule scores are graded (0 / 0.35 / 0.7 / 1.0) and
    the teacher emits probabilities, so both are continuous, while
    roc_auc_score accepts hard labels only - it raises "continuous format is
    not supported". Anything outside {0, 1} is therefore binarised at
    `threshold`, which is the intended reading when scoring a model against
    soft targets: did it rank the cells the target calls positive above the
    rest. Step 4 validates against the teacher's probabilities and would hit
    exactly the same wall, so this belongs in the metric, not at each call
    site.

    Columns that are constant, or that have fewer than two usable rows, are
    undefined and reported as nan rather than silently averaged in.
    """
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true, dtype=np.float64)
    if not np.isin(y_true, (0.0, 1.0)).all():
        y_true = (y_true > threshold).astype(np.float64)
    per: Dict[str, float] = {}
    for i, name in enumerate(LABELS):
        sel = np.ones(len(y_true), dtype=bool) if mask is None else (mask[:, i] > 0)
        yt, yp = y_true[sel, i], y_pred[sel, i]
        if len(yt) < 2 or yt.min() == yt.max():
            per[name] = float("nan")
            continue
        per[name] = float(roc_auc_score(yt, yp))
    vals = [v for v in per.values() if not math.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def print_auc_table(macro: float, per: Dict[str, float]) -> None:
    print(f"  macro AUC = {macro:.5f}")
    for k, v in per.items():
        bar = "#" * int(max(v - 0.5, 0) * 60) if not math.isnan(v) else ""
        print(f"    {k:<18s} {v:.4f} {bar}")


def rank_normalise(x: np.ndarray) -> np.ndarray:
    """Per-column rank transform to [0,1]; AUC is rank-based so this is lossless
    for a single model and makes ensembling of differently-calibrated models safe."""
    out = np.empty_like(x, dtype=np.float64)
    for j in range(x.shape[1]):
        order = np.argsort(np.argsort(x[:, j]))
        out[:, j] = order / max(len(order) - 1, 1)
    return out
