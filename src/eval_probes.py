"""Measure what an embedding contains, by training a linear classifier on top of it.

A contrastive run has no accuracy of its own: the loss falls, and that says nothing
about whether the embedding is useful. The standard answer is a linear probe — freeze
the encoder, fit the simplest possible classifier on its output, and report how well it
does. Linear on purpose: anything stronger would measure the probe rather than the
representation.

    python src/eval_probes.py experiment=<name> model=simclr \\
        ckpt_path=logs/train/runs/<date>/checkpoints/<file>.ckpt

It writes `probe_results.json` into the run directory and prints a table. Compare the
number against a classifier trained end to end on the same data, `model=tinyTransformer`:
a probe that matches it means the embedding kept what the task needs.

The embeddings themselves are written next to the results when `--save-embeddings` is
passed, which is what a UMAP plot or an anomaly-detection study starts from.
"""

import json
import os
from typing import Dict, Tuple

import hydra
import rootutils
import torch
import torch.nn as nn
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


@torch.no_grad()
def extract_embeddings(
    model: LightningModule, loader: DataLoader, device: str, limit: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen encoder over a loader and keep what it produces.

    `limit` caps the number of events, which is what makes this usable on a train split
    of millions: a probe saturates long before that.
    """
    model.eval()
    embeddings, labels = [], []
    seen = 0
    for x, y in loader:
        out = model.get_embeddings(x.to(device)).float().cpu()
        embeddings.append(out)
        labels.append(y)
        seen += len(y)
        if limit is not None and seen >= limit:
            break
    X = torch.cat(embeddings)[:limit]
    y = torch.cat(labels)[:limit]
    log.info(f"{len(y):,} events, embeddings of dimension {X.shape[1]}")
    return X, y


def train_linear_probe(
    train: Tuple[torch.Tensor, torch.Tensor],
    val: Tuple[torch.Tensor, torch.Tensor],
    num_classes: int,
    epochs: int = 30,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    batch_size: int = 1024,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Fit `num_classes` logistic regressions on frozen embeddings.

    Standardising the features first is not cosmetic: without it the probe's learning
    rate has to be tuned per encoder, and a comparison between two encoders turns into a
    comparison of their scales.
    """
    torch.manual_seed(seed)
    X_train, y_train = train
    X_val, y_val = val

    mean, std = X_train.mean(0, keepdim=True), X_train.std(0, keepdim=True).clamp_min(1e-6)
    X_train, X_val = (X_train - mean) / std, (X_val - mean) / std

    probe = nn.Linear(X_train.shape[1], num_classes).to(device)
    optimiser = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
    # How much optimisation this actually is. A probe on few events with a large batch
    # takes one step per epoch, which underfits and reads as a poor embedding: lr 1e-3
    # with 800 events and a batch of 1024 reached 0.34 accuracy on data a probe can
    # separate perfectly.
    log.info(f"  {len(loader)} steps per epoch, {epochs * len(loader)} in total, lr {lr}")
    history = {}
    for epoch in range(epochs):
        probe.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(yb)

        probe.eval()
        with torch.no_grad():
            val_acc = accuracy(probe(X_val.to(device)), y_val.to(device)).item()
        history = {"train_loss": total / len(y_train), "val_acc": val_acc}
        if epoch % 5 == 0 or epoch == epochs - 1:
            log.info(f"  epoch {epoch:3d}  loss {history['train_loss']:.4f}  val acc {val_acc:.4f}")

    probe.normalisation = (mean, std)
    return probe, history


@torch.no_grad()
def score(probe: nn.Module, data: Tuple[torch.Tensor, torch.Tensor], num_classes: int,
          device: str = "cpu") -> Dict[str, float]:
    """Accuracy and macro AUROC of the probe on one split."""
    X, y = data
    mean, std = probe.normalisation
    logits = probe(((X - mean) / std).to(device))
    y = y.to(device)
    return {
        "accuracy": MulticlassAccuracy(num_classes=num_classes).to(device)(logits, y).item(),
        "auroc": MulticlassAUROC(num_classes=num_classes).to(device)(logits, y).item(),
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    if not cfg.get("ckpt_path"):
        raise ValueError("Pass the checkpoint to probe: ckpt_path=<path to .ckpt>")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_cfg = cfg.get("probe", {})
    train_limit = probe_cfg.get("train_events", 100_000)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("test")

    log.info(f"Loading <{cfg.model._target_}> from {cfg.ckpt_path}")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    # The encoder is built in setup() from the datamodule, so the shapes have to exist
    # before the checkpoint can be loaded into them.
    model.trainer = None
    model._trainer = type("T", (), {"datamodule": datamodule})()
    model.setup("test")
    state = torch.load(cfg.ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    if missing:
        log.warning(f"Not in the checkpoint: {missing}")
    model = model.to(device)

    log.info("Extracting embeddings")
    splits = {
        "train": extract_embeddings(model, datamodule.train_dataloader(), device, train_limit),
        "val": extract_embeddings(model, datamodule.val_dataloader(), device),
        "test": extract_embeddings(model, datamodule.test_dataloader(), device),
    }

    log.info("Training the linear probe")
    probe, history = train_linear_probe(
        splits["train"], splits["val"], datamodule.num_classes,
        epochs=probe_cfg.get("epochs", 30),
        lr=probe_cfg.get("lr", 1e-2),
        batch_size=probe_cfg.get("batch_size", 1024),
        weight_decay=probe_cfg.get("weight_decay", 0.0),
        device=device,
        seed=cfg.get("seed", 42),
    )

    results = {split: score(probe, data, datamodule.num_classes, device)
               for split, data in splits.items()}
    results["_meta"] = {
        "ckpt_path": str(cfg.ckpt_path),
        "model": cfg.model._target_,
        "label": cfg.data.label,
        "classes": list(cfg.data.to_classify),
        "train_events": len(splits["train"][1]),
        "embedding_dim": splits["train"][0].shape[1],
        "probe_epochs": probe_cfg.get("epochs", 30),
    }

    log.info("Linear probe on frozen embeddings:")
    for split in ("train", "val", "test"):
        log.info(f"  {split:5s} accuracy {results[split]['accuracy']:.4f}  "
                 f"AUROC {results[split]['auroc']:.4f}")

    out_dir = cfg.paths.output_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Written to {os.path.join(out_dir, 'probe_results.json')}")

    if probe_cfg.get("save_embeddings", False):
        path = os.path.join(out_dir, "embeddings.pt")
        torch.save({split: {"X": X, "y": y} for split, (X, y) in splits.items()}, path)
        log.info(f"Embeddings written to {path}")


if __name__ == "__main__":
    main()
