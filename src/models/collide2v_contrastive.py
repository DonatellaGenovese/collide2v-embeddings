"""Contrastive training of the event encoder: SimCLR and SupCon, in one module.

The two objectives share everything — the encoder, the projection head, the
augmentations, the training loop — and differ in which pairs of a batch they treat as
positive. So `positives` is a config value, not a second file:

    positives: views    two augmented views of one event are positives, and nothing
                        else is. No labels are used: this is SimCLR.
    positives: labels   events of the same process are positives, views included.
                        This is SupCon, from Khosla et al. (2020).

What comes out is the encoder. The projection head is scaffolding and is dropped
afterwards; the optional classification head is there to watch training, not to be used.

Adding a third objective means writing what your positives are, and if that cannot be
said as a grouping, a loss class next to ContrastiveLoss and one branch here.

    python src/train.py experiment=<name> model=simclr
    python src/train.py experiment=<name> model=supcon
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from .components.contrastive import ContrastiveLoss, ProjectionHead, RandomMaskingAugmentation
from .components.transformer import TinyTransformer

POSITIVES = ("views", "labels")


class COLLIDE2VContrastiveLitModule(LightningModule):
    """Train the encoder so that similar events land close together.

    Args:
        positives: "views" for SimCLR, "labels" for SupCon
        d_model, n_heads, num_layers, d_ff, dropout: the encoder, a TinyTransformer
        projection_dim, hidden_projection_dim: the head the loss sees
        temperature, base_temperature: see ContrastiveLoss
        num_views: augmented views per event. Two is the usual choice; more means a
            larger effective batch and more memory.
        augmentation: "masking" or "physics"
        mask_probability, mask_full_particle: for "masking"
        eta_sigma, pt_sigma: for "physics"
        use_classification_head: train a linear head on the embedding alongside the
            contrastive loss, to have an accuracy to watch. It is a diagnostic: the
            head is not what the embedding is for, and `classification_weight` keeps
            its gradient small.
        classification_weight: weight of that auxiliary loss
        allow_single_class_batches: with "labels", a batch holding one class has no
            negatives. True returns zero for that batch, False stops the run.
    """

    def __init__(
        self,
        positives: str = "labels",
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        projection_dim: int = 128,
        hidden_projection_dim: int = 256,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        num_views: int = 2,
        augmentation: str = "masking",
        mask_probability: float = 0.2,
        mask_full_particle: bool = False,
        eta_sigma: float = 1.5,
        pt_sigma: float = 0.1,
        use_classification_head: bool = True,
        classification_weight: float = 0.1,
        allow_single_class_batches: bool = True,
        optimizer: Optional[Callable] = None,
        scheduler: Optional[Callable] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        if positives not in POSITIVES:
            raise ValueError(f"positives must be one of {POSITIVES}, got {positives!r}.")
        if positives == "views" and num_views < 2:
            raise ValueError(
                "With positives='views' an event's only positive is its other view, so "
                "num_views must be at least 2."
            )
        if augmentation not in ("masking", "physics"):
            raise ValueError("augmentation must be 'masking' or 'physics'.")

        # Built in setup(), once the datamodule can say how wide an event is and how
        # many classes there are.
        self.encoder: Optional[nn.Module] = None
        self.projection_head: Optional[nn.Module] = None
        self.classification_head: Optional[nn.Module] = None
        self.augmentation: Optional[Callable] = None

        self.criterion = ContrastiveLoss(temperature=temperature, base_temperature=base_temperature)
        self.classification_criterion = nn.CrossEntropyLoss() if use_classification_head else None

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        if use_classification_head:
            self.val_acc_best = MaxMetric()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        """Build the model from what the datamodule knows. Called on every process."""
        if self.encoder is not None:
            return
        # The private attribute, not self.trainer, which raises Lightning's own error
        # about not being attached to a Trainer before this check can run.
        trainer = getattr(self, "_trainer", None)
        dm = getattr(trainer, "datamodule", None) if trainer else None
        if dm is None:
            raise RuntimeError(
                "No datamodule: this module reads the feature map and the number of "
                "classes from it, so it cannot be built on its own."
            )

        preproc_dir = dm.paths["eos_preproc_dir"]
        feature_map_path = os.path.join(preproc_dir, "feature_map.json")
        with open(feature_map_path) as f:
            import json

            feature_map = json.load(f)
        num_classes = dm.num_classes

        self.augmentation = self._build_augmentation(feature_map, preproc_dir)
        self.encoder = TinyTransformer(
            feature_map=feature_map,
            d_model=self.hparams.d_model,
            n_heads=self.hparams.n_heads,
            num_layers=self.hparams.num_layers,
            d_ff=self.hparams.d_ff,
            dropout=self.hparams.dropout,
            num_classes=num_classes,
        )
        self.projection_head = ProjectionHead(
            input_dim=self.hparams.d_model,
            hidden_dim=self.hparams.hidden_projection_dim,
            output_dim=self.hparams.projection_dim,
        )
        if self.hparams.use_classification_head:
            self.classification_head = nn.Linear(self.hparams.d_model, num_classes)
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        if self.hparams.compile and stage == "fit":
            self.encoder = torch.compile(self.encoder)

    def _build_augmentation(self, feature_map: Dict[str, Any], preproc_dir: str) -> Callable:
        if self.hparams.augmentation == "physics":
            from src.data.augmentations import PhysicsAugmentation

            return PhysicsAugmentation(
                feature_map_path=os.path.join(preproc_dir, "feature_map.json"),
                norm_stats_path=os.path.join(preproc_dir, "norm_stats.json"),
                eta_sigma=self.hparams.eta_sigma,
                pt_sigma=self.hparams.pt_sigma,
            )
        return RandomMaskingAugmentation(
            feature_map=feature_map,
            mask_probability=self.hparams.mask_probability,
            mask_full_particle=self.hparams.mask_full_particle,
        )

    # ------------------------------------------------------------------
    # Forward and loss
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """(embeddings, normalised projections, logits or None)."""
        embeddings = self.encoder.get_embeddings(x)
        projections = F.normalize(self.projection_head(embeddings), dim=-1, p=2)
        logits = self.classification_head(embeddings) if self.classification_head else None
        return embeddings, projections, logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """What this training produces, for probes and for anomaly detection."""
        return self.encoder.get_embeddings(x)

    def group_ids(self, labels: torch.Tensor, batch_size: int, n_views: int) -> torch.Tensor:
        """Which samples the loss should pull together, as one id per row.

        The whole difference between the two objectives. With "views", an event is its
        own group, so its views are positives and every other event is a negative — no
        labels involved. With "labels", the group is the process.
        """
        if self.hparams.positives == "views":
            return torch.arange(batch_size, device=labels.device).repeat(n_views)
        return labels.repeat(n_views)

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        num_views: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        x, labels = batch
        batch_size = x.shape[0]
        n_views = self.hparams.num_views if num_views is None else num_views

        # Augment only while training: a validation view is the event as it is, so the
        # number on the progress bar is not itself noisy.
        views = [self.augmentation(x) if self.training else x for _ in range(n_views)]
        stacked = torch.cat(views, dim=0)

        embeddings, projections, logits = self.forward(stacked)
        ids = self.group_ids(labels, batch_size, n_views)

        single_class = torch.unique(labels).numel() < 2
        if (
            self.hparams.positives == "labels"
            and single_class
            and self.hparams.allow_single_class_batches
        ):
            loss = projections.new_zeros((), requires_grad=True)
        else:
            loss = self.criterion(projections, ids, require_positives=self.training)

        result = {
            "contrastive_loss": loss,
            # First view only: these are for logging, and the views are interchangeable.
            "embeddings": embeddings[:batch_size],
            "projections": projections[:batch_size],
        }

        if logits is not None:
            first_view = logits[:batch_size]
            result["classification_loss"] = self.classification_criterion(first_view, labels)
            result["preds"] = first_view.argmax(dim=-1)
        return result

    def _total_loss(self, result: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = result["contrastive_loss"]
        if "classification_loss" in result:
            loss = loss + self.hparams.classification_weight * result["classification_loss"]
        return loss

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        result = self.model_step(batch)
        loss = self._total_loss(result)

        self.train_loss(result["contrastive_loss"])
        self.log("train/contrastive_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        if "preds" in result:
            self.train_acc(result["preds"], batch[1])
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
            self.log("train/classification_loss", result["classification_loss"], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        result = self.model_step(batch)
        self.val_loss(result["contrastive_loss"])
        self.log("val/contrastive_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if "preds" in result:
            self.val_acc(result["preds"], batch[1])
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        result = self.model_step(batch)
        self.log("test/contrastive_loss", result["contrastive_loss"], on_step=False, on_epoch=True)
        if "preds" in result:
            self.log("test/acc", self.val_acc(result["preds"], batch[1]), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.use_classification_head:
            self.val_acc_best(self.val_acc.compute())
            self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/contrastive_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
