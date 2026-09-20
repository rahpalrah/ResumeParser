"""
knee_data.py - study-level dataset built on the sprite cache.

One item = one study = a fixed stack of `N_SLOTS` series, each `n_slices` deep,
each slice turned into a 3-channel 2.5D image from its immediate neighbours.
Missing sequences are zero-filled and masked, so protocol variation across the
international sites never changes the tensor shape.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from knee_common import (LABELS, N_LABELS, N_SLOTS, PLANE2IDX, SLOTS, Cfg,
                         read_sprite, sprite_path)

IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226


def _augment_volume(vol: np.ndarray, rng: random.Random) -> np.ndarray:
    """Geometric + photometric augmentation applied identically to every slice
    of a series (a per-slice warp would destroy 2.5D continuity)."""
    k, h, w = vol.shape
    angle = rng.uniform(-12, 12)
    scale = rng.uniform(0.9, 1.12)
    tx = rng.uniform(-0.06, 0.06) * w
    ty = rng.uniform(-0.06, 0.06) * h
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    out = np.empty_like(vol)
    for i in range(k):
        out[i] = cv2.warpAffine(vol[i], m, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.5:
        gain = rng.uniform(0.85, 1.15)
        bias = rng.uniform(-18, 18)
        out = np.clip(out.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:                       # slice dropout
        j = rng.randrange(k)
        out[j] = 0
    return out


def _to_25d(vol: np.ndarray) -> np.ndarray:
    """(K,H,W) uint8 -> (K,3,H,W) float32, channels = neighbouring slices."""
    k = vol.shape[0]
    prev = np.concatenate([vol[:1], vol[:-1]], axis=0)
    nxt = np.concatenate([vol[1:], vol[-1:]], axis=0)
    x = np.stack([prev, vol, nxt], axis=1).astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class KneeStudyDataset(Dataset):
    def __init__(self, studies: pd.DataFrame, series: pd.DataFrame, cfg: Cfg,
                 train: bool = True, targets: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None):
        """
        studies : DataFrame with at least StudyInstanceUID (and label columns
                  when supervised); one row per study.
        series  : the *_series.csv frame, already carrying a `slot` column and
                  a `Laterality` column from the preprocessing step.
        """
        self.cfg = cfg
        self.train = train
        self.studies = studies.reset_index(drop=True)
        self.uids: List[str] = self.studies["StudyInstanceUID"].tolist()
        self.targets = targets
        self.weights = weights if weights is not None else np.ones(len(self.uids), np.float32)

        s = series[series["slot"] >= 0]
        self.by_study: Dict[str, List[dict]] = {
            uid: grp.to_dict("records") for uid, grp in s.groupby("StudyInstanceUID")
        }
        self.lat_by_study: Dict[str, int] = {}
        if "Laterality" in series.columns:
            for uid, grp in series.groupby("StudyInstanceUID"):
                vals = [v for v in grp["Laterality"].tolist() if v in (0, 1)]
                self.lat_by_study[uid] = int(vals[0]) if vals else 2

    def __len__(self) -> int:
        return len(self.uids)

    def _load_slot(self, uid: str, recs: List[dict], slot: int, rng: random.Random):
        cands = [r for r in recs if int(r["slot"]) == slot]
        if not cands:
            return None
        rec = rng.choice(cands) if (self.train and len(cands) > 1) else cands[0]
        vol = read_sprite(sprite_path(self.cfg.cache_dir, uid, rec["SeriesInstanceUID"]),
                          self.cfg.n_slices, self.cfg.img_size, self.cfg.grid_w)
        if vol is None:
            return None
        return vol, rec

    def __getitem__(self, i: int):
        cfg = self.cfg
        uid = self.uids[i]
        rng = random.Random((hash(uid) ^ random.getrandbits(32)) if self.train else hash(uid))
        recs = self.by_study.get(uid, [])

        S, K, Z = N_SLOTS, cfg.n_slices, cfg.img_size
        image = np.zeros((S, K, 3, Z, Z), np.float32)
        slot = np.zeros(S, np.int64)
        plane = np.full(S, len(PLANE2IDX), np.int64)
        fluid = np.full(S, 2, np.int64)
        fat = np.full(S, 2, np.int64)
        mask = np.zeros(S, np.int64)

        lat = self.lat_by_study.get(uid, 2)
        flip = self.train and rng.random() < 0.5
        if flip and lat in (0, 1):
            lat = 1 - lat                     # mirroring turns a left knee into a right one
        elif flip:
            lat = 2

        for s in range(S):
            got = self._load_slot(uid, recs, s, rng)
            slot[s] = s
            if got is None:
                continue
            vol, rec = got
            if self.train:
                vol = _augment_volume(vol, rng)
            if flip:
                vol = vol[:, :, ::-1].copy()
            image[s] = _to_25d(vol)
            plane[s] = PLANE2IDX.get(str(rec.get("Anatomical_Plane", "")), len(PLANE2IDX))
            fluid[s] = int(rec.get("Fluid_Sensitive", 2)) if rec.get("Fluid_Sensitive") in (0, 1) else 2
            fat[s] = int(rec.get("Fat_Suppression", 2)) if rec.get("Fat_Suppression") in (0, 1) else 2
            mask[s] = 1

        item = {
            "image": torch.from_numpy(image),
            "slot": torch.from_numpy(slot),
            "plane": torch.from_numpy(plane),
            "fluid": torch.from_numpy(fluid),
            "fat": torch.from_numpy(fat),
            "series_mask": torch.from_numpy(mask),
            "lat": torch.tensor(lat, dtype=torch.long),
            "weight": torch.tensor(float(self.weights[i]), dtype=torch.float32),
            "uid": uid,
        }
        if self.targets is not None:
            item["target"] = torch.from_numpy(self.targets[i].astype(np.float32))
        return item


def collate(batch: List[dict]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k in batch[0]:
        if k == "uid":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def make_folds(df: pd.DataFrame, n_folds: int, seed: int = 42) -> np.ndarray:
    """Multi-label stratified folds, falling back to plain KFold if the
    iterative-stratification package is unavailable (it is pip-only)."""
    y = df[LABELS].values
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = np.zeros(len(df), np.int64)
        for f, (_, va) in enumerate(splitter.split(df, y)):
            folds[va] = f
        return folds
    except Exception:
        from sklearn.model_selection import KFold
        folds = np.zeros(len(df), np.int64)
        for f, (_, va) in enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(df)):
            folds[va] = f
        return folds
