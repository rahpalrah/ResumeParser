"""
knee_model.py - CARE-Net: Compartment-Aware, Report-distilled Ensemble network.

The architecture exists to exploit three things this competition gives you that
a generic multi-label CNN ignores:

1. `train.csv` ships a free-text report for *every* study but binary labels for
   only a small subset.  A text teacher reads the reports and distils soft
   targets onto the unlabeled majority (handled in step 2; this file just
   accepts soft targets in the loss).

2. `*_series.csv` ships the plane / fluid-sensitivity / fat-suppression of every
   series *at test time too*.  So the study head can be told what it is looking
   at instead of having to infer it: each series token carries learned
   embeddings for its slot, plane, fluid and fat flags.

3. Five of the twelve targets are compartment-specific (medial vs lateral).  In
   a coronal or axial slice those compartments are the left and right halves of
   the image, modulo knee laterality.  The compartment branch pools the two
   halves separately and routes them to the compartment-specific labels, with a
   laterality token telling the model which half is medial.

Output coupling: the twelve findings are strongly correlated (an ACL tear comes
with effusion and lateral-compartment contusion far more often than chance).  A
zero-initialised low-rank coupling term lets the head move correlated logits
together without destabilising early training.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from knee_common import LABELS, MEDIAL_LABELS, LATERAL_LABELS, N_LABELS, N_SLOTS, PLANES

MEDIAL_IDX = [LABELS.index(x) for x in MEDIAL_LABELS]
LATERAL_IDX = [LABELS.index(x) for x in LATERAL_LABELS]


def sinusoidal_positions(n: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(dim // 2, device=device, dtype=torch.float32).unsqueeze(0)
    ang = pos / torch.pow(10000.0, (2 * i) / dim)
    out = torch.zeros(n, dim, device=device)
    out[:, 0::2] = torch.sin(ang)
    out[:, 1::2] = torch.cos(ang)
    return out


class TransformerStack(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, drop: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=drop, activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.enc(x, src_key_padding_mask=pad_mask)


class AttnPool(nn.Module):
    """Masked attention pooling - one learned query over a token sequence."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = self.score(x)                                   # B,T,1
        if mask is not None:
            w = w.masked_fill(mask.unsqueeze(-1), -1e4)
        w = torch.softmax(w, dim=1)
        return (w * x).sum(dim=1)


class CoupledHead(nn.Module):
    """Linear head plus a zero-initialised low-rank label-coupling residual."""

    def __init__(self, dim: int, n_labels: int = N_LABELS, rank: int = 4, enabled: bool = True):
        super().__init__()
        self.base = nn.Linear(dim, n_labels)
        self.enabled = enabled
        if enabled:
            self.u = nn.Parameter(torch.zeros(n_labels, rank))
            self.v = nn.Parameter(torch.randn(rank, n_labels) * 0.02)
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.base(h)
        if self.enabled:
            z = z + torch.tanh(self.gate) * (torch.tanh(z) @ self.u @ self.v)
        return z


class CareNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm

        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=cfg.pretrained, num_classes=0,
            in_chans=3, drop_path_rate=cfg.drop_path,
        )
        feat_dim = self.backbone.num_features
        d = cfg.embed_dim

        self.proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d))
        self.slice_tf = TransformerStack(d, cfg.slice_layers, cfg.n_heads, cfg.drop)
        self.slice_pool = AttnPool(d)

        # series conditioning: slot, plane, fluid-sensitive, fat-suppression
        self.emb_slot = nn.Embedding(N_SLOTS + 1, d)
        self.emb_plane = nn.Embedding(len(PLANES) + 1, d)
        self.emb_fluid = nn.Embedding(3, d)
        self.emb_fat = nn.Embedding(3, d)
        self.emb_lat = nn.Embedding(3, d)          # 0=left 1=right 2=unknown

        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.study_tf = TransformerStack(d, cfg.study_layers, cfg.n_heads, cfg.drop)

        self.norm = nn.LayerNorm(d)
        self.head = CoupledHead(d, N_LABELS, enabled=cfg.use_coupling)

        # compartment branch: separate medial / lateral pooled features
        self.use_comp = cfg.use_compartment
        if self.use_comp:
            self.comp_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d))
            self.comp_pool = AttnPool(d)
            self.head_med = nn.Linear(d, len(MEDIAL_IDX))
            self.head_lat = nn.Linear(d, len(LATERAL_IDX))
            self.comp_mix = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------ #
    def _encode_slices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (N,3,H,W) -> pooled (N,F) and half-split features (N,2,F)."""
        fmap = self.backbone.forward_features(x)            # N,C,h,w  (or N,T,C)
        if fmap.dim() == 3:                                 # ViT-style tokens
            n, t, c = fmap.shape
            s = int(math.sqrt(t))
            if s * s == t:
                fmap = fmap.transpose(1, 2).reshape(n, c, s, s)
            else:                                           # has a CLS token
                fmap = fmap[:, 1:].transpose(1, 2)
                s = int(math.sqrt(fmap.shape[-1]))
                fmap = fmap.reshape(n, c, s, s)
        if fmap.shape[1] < fmap.shape[-1] and fmap.dim() == 4 and fmap.shape[-1] == fmap.shape[-2]:
            pass                                            # already N,C,h,w
        pooled = fmap.mean(dim=(2, 3))
        w = fmap.shape[-1]
        left = fmap[..., : w // 2].mean(dim=(2, 3))
        right = fmap[..., w - w // 2:].mean(dim=(2, 3))
        return pooled, torch.stack([left, right], dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img = batch["image"]                                # B,S,K,3,H,W
        B, S, K = img.shape[:3]
        flat = img.reshape(B * S * K, *img.shape[3:])

        pooled, halves = self._encode_slices(flat)
        d_feat = pooled.shape[-1]

        # ---- slice level -------------------------------------------------
        tok = self.proj(pooled).reshape(B * S, K, -1)
        tok = tok + sinusoidal_positions(K, tok.shape[-1], tok.device).unsqueeze(0)
        tok = self.slice_tf(tok)
        series = self.slice_pool(tok).reshape(B, S, -1)      # B,S,d

        # ---- series conditioning ----------------------------------------
        series = (series
                  + self.emb_slot(batch["slot"])
                  + self.emb_plane(batch["plane"])
                  + self.emb_fluid(batch["fluid"])
                  + self.emb_fat(batch["fat"]))

        pad = batch["series_mask"] == 0                      # True where absent
        cls = self.cls.expand(B, -1, -1) + self.emb_lat(batch["lat"]).unsqueeze(1)
        seq = torch.cat([cls, series], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], dim=1)
        seq = self.study_tf(seq, pad_mask=pad)
        h = self.norm(seq[:, 0])

        logits = self.head(h)

        # ---- compartment branch -----------------------------------------
        out: Dict[str, torch.Tensor] = {}
        if self.use_comp:
            hv = self.comp_proj(halves.reshape(-1, d_feat)).reshape(B, S, K, 2, -1)
            # laterality: for a left knee the medial compartment sits on the
            # image-right half, for a right knee on the image-left half.
            lat = batch["lat"].view(B, 1, 1, 1)
            flip = (lat == 0)                                # left knee -> swap
            # flip is (B,1,1,1) and each half is (B,S,K,d): broadcasting lines
            # the study-level decision up with every slice of every series.
            hv_med = torch.where(flip, hv[:, :, :, 1], hv[:, :, :, 0])
            hv_lat = torch.where(flip, hv[:, :, :, 0], hv[:, :, :, 1])

            # only coronal / axial series carry a meaningful medial-lateral axis
            coronal_axial = (batch["plane"] != 1).float()    # plane idx 0=Sagittal
            wmask = (batch["series_mask"].float() * coronal_axial).unsqueeze(-1).unsqueeze(-1)
            def _pool(v):
                v = (v * wmask)
                denom = wmask.sum(dim=(1, 2)).clamp(min=1e-3)
                return v.sum(dim=(1, 2)) / denom
            f_med, f_lat = _pool(hv_med), _pool(hv_lat)
            out["logit_med"] = self.head_med(f_med)
            out["logit_lat"] = self.head_lat(f_lat)

            # The branch is only meaningful when the knee side is known. With an
            # unknown side there is no defensible mapping from image half to
            # compartment, and the previous code silently took flip=False -
            # calling the image-left half medial for every one of them. About
            # half of those are left knees, so it was training the medial head
            # on lateral anatomy for a quarter of the corpus: a wrong signal,
            # not a missing one, and precisely what collapses Medial OA and
            # Lateral OA into the same predictor.
            known = (batch["lat"] != 2).float().unsqueeze(-1)      # (B,1)
            out["comp_valid"] = known.squeeze(-1)
            mix = torch.sigmoid(self.comp_mix) * known
            adj = logits.clone()
            adj[:, MEDIAL_IDX] = ((1 - mix) * logits[:, MEDIAL_IDX]
                                  + mix * out["logit_med"])
            adj[:, LATERAL_IDX] = ((1 - mix) * logits[:, LATERAL_IDX]
                                   + mix * out["logit_lat"])
            logits = adj

        out["logits"] = logits
        out["embed"] = h
        return out


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

class SoftAsymmetricLoss(nn.Module):
    """BCE that accepts soft targets, with optional asymmetric focusing.

    Soft targets are required because the report teacher emits probabilities,
    not 0/1.  Asymmetric focusing (gamma_neg > gamma_pos) matters because the
    rarer findings - Fracture, Synovitis - are swamped by negatives.
    """

    def __init__(self, gamma_neg: float = 2.0, gamma_pos: float = 0.0,
                 clip: float = 0.02, label_smooth: float = 0.005):
        super().__init__()
        self.gn, self.gp, self.clip, self.ls = gamma_neg, gamma_pos, clip, label_smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        t = target.clamp(self.ls, 1.0 - self.ls)
        p = torch.sigmoid(logits)
        p_neg = (1 - p + self.clip).clamp(max=1.0)
        loss_pos = t * torch.log(p.clamp(min=1e-8)) * torch.pow(1 - p, self.gp)
        loss_neg = (1 - t) * torch.log(p_neg.clamp(min=1e-8)) * torch.pow(p, self.gn)
        loss = -(loss_pos + loss_neg)
        if weight is not None:
            # (B,) weights one study uniformly; (B,L) weights each cell, which
            # is what sparse gold labels need - a study can carry a real
            # annotation for Effusion and only a teacher guess for Fracture.
            loss = loss * (weight if weight.dim() == 2 else weight.view(-1, 1))
        return loss.mean()


class ModelEMA:
    """Exponential moving average of weights - cheap and reliably worth ~0.003
    macro AUC on noisy multi-label medical targets."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])
