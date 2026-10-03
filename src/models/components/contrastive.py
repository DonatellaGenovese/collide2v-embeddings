"""The pieces a contrastive objective is built from: a head, a loss, an augmentation.

SimCLR and SupCon differ in one thing only — which pairs in a batch count as positive —
so there is one loss here, and the caller decides. That is also what makes a third
objective cheap to add: write what your positives are, and reuse the rest.
"""

from typing import Any, Dict

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Maps an embedding to the space the contrastive loss works in.

    It exists so that the loss does not shape the embedding directly: what is kept
    afterwards is the encoder output, and this head is thrown away. Two linear layers
    with a LayerNorm between them, no activation at the end, because the output is L2
    normalised before the loss sees it.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveLoss(nn.Module):
    """Pull positives together and push everything else apart, in one formula.

    For each sample it maximises the similarity to its positives against the similarity
    to the whole batch, which is the loss of Khosla et al. (2020). `group_ids` says what
    a positive is, and that single choice is the difference between the two objectives
    this repository ships:

        SupCon  — group_ids are the class labels, so every event of the same process
                  is a positive. The embedding is organised by process.
        SimCLR  — group_ids identify the event, so the only positive of a view is the
                  other view of the same event. No labels are used at all.

    `temperature` sharpens the distribution: lower values put more weight on the hardest
    negatives. `base_temperature` only rescales the gradient, and is kept because the
    published runs set both to 0.07.
    """

    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        group_ids: torch.Tensor,
        require_positives: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            features: L2-normalised projections, [N, dim], N = batch x views
            group_ids: [N], equal values mark a positive pair
            require_positives: raise when the batch holds a single group, instead of
                returning zero. True while training, where a silent zero loss would
                look like convergence; False for a validation batch that happens to be
                degenerate.
        """
        if features.shape[0] < 2:
            raise ValueError("ContrastiveLoss needs at least 2 samples.")

        ids = group_ids.contiguous().view(-1, 1)
        if ids.shape[0] != features.shape[0]:
            raise ValueError(
                f"{features.shape[0]} features against {ids.shape[0]} group ids."
            )

        if torch.unique(ids).numel() < 2:
            if require_positives:
                raise ValueError(
                    "Every sample in this batch belongs to the same group, so the loss "
                    "has no negatives and nothing to learn from. With class labels as "
                    "groups this means a single-class batch: shuffle more, or raise the "
                    "batch size."
                )
            return features.new_zeros((), requires_grad=True)

        # Cosine similarity, since the features are normalised. The row maximum is
        # subtracted for numerical stability; it cancels in the ratio below.
        similarity = features @ features.T
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
        logits = similarity / self.temperature

        off_diagonal = torch.ones_like(logits).fill_diagonal_(0)
        positives = torch.eq(ids, ids.T).float() * off_diagonal

        # log of the softmax over everything but the sample itself
        log_prob = logits - torch.log((logits.exp() * off_diagonal).sum(1, keepdim=True) + 1e-9)

        # Average over each sample's positives. A sample with none contributes zero
        # rather than a division by zero; with views as groups that cannot happen, with
        # labels it can, when a class appears exactly once in the batch.
        positives_per_sample = positives.sum(1)
        divisor = torch.where(positives_per_sample == 0,
                              torch.ones_like(positives_per_sample),
                              positives_per_sample)
        mean_log_prob = (positives * log_prob).sum(1) / divisor

        return -(self.temperature / self.base_temperature) * mean_log_prob.mean()


class RandomMaskingAugmentation(nn.Module):
    """Drops parts of an event, so that two views of it are not identical.

    Standing in for a detector that does not see everything: an inefficiency, a particle
    below threshold, an object lost in reconstruction. Masking means setting to zero,
    which is what padding already is in these vectors, so a masked slot is
    indistinguishable from an object that was never there.

    Args:
        feature_map: the expanded feature map, which says where each object's slots are
        mask_probability: chance of dropping a slot, or an object when
            `mask_full_particle`
        mask_full_particle: drop whole objects instead of single features. Closer to a
            detector failure; single features are a noisier, weaker augmentation.
    """

    def __init__(
        self,
        feature_map: Dict[str, Any],
        mask_probability: float = 0.2,
        mask_full_particle: bool = False,
    ):
        super().__init__()
        self.mask_probability = mask_probability
        self.mask_full_particle = mask_full_particle

        # Only object groups can lose an object; a scalar such as MET has no slots.
        self.object_groups = [
            {
                "start": cfg["start"],
                "num_objects": cfg["topk"],
                "columns_per_object": len(cfg["columns"]),
            }
            for cfg in feature_map.values()
            if cfg.get("topk") is not None
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        augmented = x.clone()
        if not self.mask_full_particle:
            drop = torch.rand_like(x) < self.mask_probability
            augmented[drop] = 0.0
            return augmented

        for group in self.object_groups:
            start = group["start"]
            n_objects = group["num_objects"]
            width = group["columns_per_object"]
            drop = torch.rand(x.shape[0], n_objects, device=x.device) < self.mask_probability
            flat = drop.unsqueeze(-1).expand(-1, -1, width).reshape(x.shape[0], -1)
            block = augmented[:, start:start + n_objects * width]
            block[flat] = 0.0
        return augmented
