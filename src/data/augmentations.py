"""Physics-motivated augmentations, applied to already normalised event vectors.

An augmentation for a contrastive objective has to change the numbers without changing
what the event is. For images one crops or recolours; for a collision there are
transformations that physics says leave the event equivalent:

  phi_rotation  the detector has no preferred azimuth, so rotating every object by the
                same Δφ gives an equally valid event. Exact here, because φ is stored as
                (sin, cos) and normalised with "none", so the pair can be rotated
                directly.
  eta_boost     a longitudinal boost shifts every η by the same amount. True of the
                collision, approximate in the detector, which is not uniform in η.
  pt_smearing   the measured pT of an object is the true value plus resolution, so
                adding noise of that size produces a measurement that could have
                happened.

These act on **normalised** vectors, the ones the loader returns, which is why they need
`norm_stats.json`: a shift of Δη in physical units is a shift of Δη/IQR on a
robust-normalised column, and the IQR is in that file. They also need the expanded
`feature_map.json`, to know which positions hold η, pT and the (sin, cos) of φ.

Padding is zero in these vectors, and an empty slot stays empty: the shifts are applied
only where the value is not zero, so an augmented event never grows an object that was
not reconstructed.

A note on what this cannot fix: an event is a flat vector of a fixed number of slots, so
anything that would reorder or add objects is out of reach here.
"""

import json
import math
import re
from typing import List, Tuple

import torch


class PhysicsAugmentation:
    """One augmented view of a normalised event, or of a batch of them.

    Args:
        feature_map_path: expanded feature_map.json, from the preprocessed directory
        norm_stats_path: norm_stats.json, from the same place
        phi_rotation, eta_boost, pt_smearing: which transformations to apply
        eta_sigma: width of Δη ~ N(0, eta_sigma), in units of η
        pt_sigma: noise in log1p(pT) space, so roughly a fractional pT resolution
    """

    def __init__(
        self,
        feature_map_path: str,
        norm_stats_path: str,
        phi_rotation: bool = True,
        eta_boost: bool = True,
        pt_smearing: bool = True,
        eta_sigma: float = 1.5,
        pt_sigma: float = 0.1,
    ):
        self.phi_rotation = phi_rotation
        self.eta_boost = eta_boost
        self.pt_smearing = pt_smearing
        self.eta_sigma = eta_sigma
        self.pt_sigma = pt_sigma

        with open(feature_map_path) as f:
            feature_map = json.load(f)
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

        block_stats = norm_stats.get("block_group_stats", {})

        self.phi_pairs: List[Tuple[int, int]] = []
        self.eta_entries: List[Tuple[int, float]] = []
        self.pt_entries: List[Tuple[int, float]] = []

        for section_name, section in feature_map.items():
            start = section["start"]
            columns = section["columns"]
            topk = section.get("topk") or 1
            n_cols = len(columns)

            phi_sin_j = next((j for j, c in enumerate(columns) if c.endswith("_Phi_sin")), None)
            phi_cos_j = next((j for j, c in enumerate(columns) if c.endswith("_Phi_cos")), None)
            eta_j = next((j for j, c in enumerate(columns) if re.search(r"_Eta$", c)), None)
            pt_j = next((j for j, c in enumerate(columns) if re.search(r"_PT$", c)), None)
            met_j = next((j for j, c in enumerate(columns) if re.search(r"_MET$", c)), None)

            # The IQR of the column, so a physical shift becomes a shift in normalised
            # units. 1.0 when the group was not normalised that way, which leaves the
            # shift in whatever units the column is already in.
            eta_iqr = float(block_stats.get(f"{section_name}.eta", {}).get("iqr", 1.0))
            pt_iqr = float(block_stats.get(f"{section_name}.pt", {}).get("iqr", 1.0))
            met_iqr = float(block_stats.get(f"{section_name}.met", {}).get("iqr", 1.0))

            for k in range(topk):
                base = start + k * n_cols
                if phi_sin_j is not None and phi_cos_j is not None:
                    self.phi_pairs.append((base + phi_sin_j, base + phi_cos_j))
                if eta_j is not None:
                    self.eta_entries.append((base + eta_j, eta_iqr))
                if pt_j is not None:
                    self.pt_entries.append((base + pt_j, pt_iqr))
                if met_j is not None:
                    # MET is smeared like a pT, with its own scale.
                    self.pt_entries.append((base + met_j, met_iqr))

        self._sin_idx = self._as_index([s for s, _ in self.phi_pairs])
        self._cos_idx = self._as_index([c for _, c in self.phi_pairs])
        self._eta_idx = self._as_index([i for i, _ in self.eta_entries])
        self._eta_iqr = self._as_scale([q for _, q in self.eta_entries])
        self._pt_idx = self._as_index([i for i, _ in self.pt_entries])
        self._pt_iqr = self._as_scale([q for _, q in self.pt_entries])

    @staticmethod
    def _as_index(values):
        return torch.tensor(values, dtype=torch.long) if values else None

    @staticmethod
    def _as_scale(values):
        return torch.tensor(values, dtype=torch.float32) if values else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """One event `[D]` or a batch `[B, D]`, augmented. The input is not modified."""
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        x = x.clone()

        if self.phi_rotation:
            x = self._rotate_phi(x)
        if self.eta_boost:
            x = self._boost_eta(x)
        if self.pt_smearing:
            x = self._smear_pt(x)

        return x.squeeze(0) if single else x

    def _rotate_phi(self, x: torch.Tensor) -> torch.Tensor:
        """Δφ ~ Uniform(0, 2π), the same for every object in an event."""
        if self._sin_idx is None:
            return x
        sin_idx, cos_idx = self._sin_idx.to(x.device), self._cos_idx.to(x.device)
        dphi = torch.empty(x.shape[0], 1, device=x.device).uniform_(0.0, 2.0 * math.pi)
        cos_d, sin_d = torch.cos(dphi), torch.sin(dphi)
        s, c = x[:, sin_idx], x[:, cos_idx]
        x[:, sin_idx] = s * cos_d + c * sin_d
        x[:, cos_idx] = c * cos_d - s * sin_d
        return x

    def _boost_eta(self, x: torch.Tensor) -> torch.Tensor:
        """Δη ~ N(0, eta_sigma), the same for every object in an event."""
        if self._eta_idx is None:
            return x
        idx, iqr = self._eta_idx.to(x.device), self._eta_iqr.to(x.device)
        deta = torch.randn(x.shape[0], 1, device=x.device) * self.eta_sigma
        values = x[:, idx]
        occupied = (values != 0.0).float()
        x[:, idx] = values + occupied * (deta / iqr.unsqueeze(0))
        return x

    def _smear_pt(self, x: torch.Tensor) -> torch.Tensor:
        """Independent noise per object, since resolution is a per-object effect."""
        if self._pt_idx is None:
            return x
        idx, iqr = self._pt_idx.to(x.device), self._pt_iqr.to(x.device)
        noise = torch.randn(x.shape[0], idx.numel(), device=x.device) * self.pt_sigma
        values = x[:, idx]
        occupied = (values != 0.0).float()
        x[:, idx] = values + occupied * (noise / iqr.unsqueeze(0))
        return x
