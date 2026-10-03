"""A minimal model that works with this pipeline. Copy this file to start your own.

It classifies events with two linear layers, which is deliberately uninteresting: what
this file demonstrates is the contract, not an architecture. Everything a model here has
to do is marked CONTRACT below, and there are five things.

    cp src/models/template_model.py src/models/my_model.py
    cp configs/model/template.yaml configs/model/my_model.yaml   # change _target_
    python src/train.py experiment=fm_testing_binary model=my_model trainer=cpu \\
        +trainer.limit_train_batches=5 +trainer.limit_val_batches=2

That last command is the check to run before anything long: five batches on CPU proves
the model builds, trains and logs, in under a minute.

Why the contract is what it is: an event vector's width and the number of classes are
properties of the *dataset*, not of the config, and they are only known once the
datamodule has prepared the data. So a model cannot build its layers in `__init__`.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class TemplateLitModule(LightningModule):
    """Two linear layers on the flat event vector, with everything the pipeline expects.

    CONTRACT 1 — take hyperparameters only, and call save_hyperparameters().
        Whatever `configs/model/<name>.yaml` holds arrives here as arguments. Nothing
        about the data is passed in; see CONTRACT 2. save_hyperparameters() puts them in
        the checkpoint, which is what lets a run be loaded back.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        optimizer: Optional[Callable] = None,
        scheduler: Optional[Callable] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # The network is None until setup(): see CONTRACT 2.
        self.net: Optional[nn.Module] = None

        self.criterion = nn.CrossEntropyLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def setup(self, stage: str) -> None:
        """CONTRACT 2 — build the layers here, from the datamodule.

        Lightning calls this after the datamodule has prepared the data and on every
        process. Two things are worth reading off it:

            dm.vlen         width of an event vector, after preprocessing
            dm.num_classes  how many processes are being classified

        A model that works on objects rather than on the flat vector reads
        `feature_map.json` from `dm.paths["eos_preproc_dir"]` instead, which says where
        each group of features sits; `TinyTransformer` does that to build its tokens.

        setup() is called more than once — fit, then validate, then test — so building
        must be idempotent, hence the guard.
        """
        if self.net is not None:
            return
        # getattr on the private attribute, not self.trainer: that property raises
        # Lightning's own "not attached to a Trainer" first, and the point here is to
        # say which piece of information is missing.
        trainer = getattr(self, "_trainer", None)
        dm = getattr(trainer, "datamodule", None) if trainer else None
        if dm is None:
            raise RuntimeError(
                "No datamodule, so the width of an event and the number of classes are "
                "unknown. This model is built by the pipeline, not on its own."
            )

        self.net = nn.Sequential(
            nn.Linear(dm.vlen, self.hparams.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_dim, dm.num_classes),
        )

        # Metrics need the class count too, so they are created here as well.
        self.train_acc = Accuracy(task="multiclass", num_classes=dm.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=dm.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=dm.num_classes)

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits for a batch of events, `[batch, vlen] -> [batch, num_classes]`."""
        return self.net(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """CONTRACT 3 — return one vector per event, before the output layer.

        Optional for a pure classifier, required for anything whose point is the
        representation: `src/eval_probes.py` calls this, and so would an
        anomaly-detection study. Here it is the hidden layer; in `TinyTransformer` it is
        the mean over tokens.
        """
        return self.net[:-1](x)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """A batch is `(x, y)`: features `[batch, vlen]` float32, labels `[batch]` long.

        The labels are positions in `data.to_classify`, in that order.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = F.softmax(logits, dim=-1).argmax(dim=-1)
        return loss, preds, y

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """CONTRACT 4 — log under the names the callbacks and configs expect.

        `train/loss`, `train/acc`, `val/loss`, `val/acc`: the checkpoint callback, early
        stopping and the Optuna sweeps all refer to these by name, so a model that logs
        something else trains fine and then cannot be checkpointed on its best epoch.
        Returning the loss is what Lightning backpropagates.
        """
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """`val/acc_best` is what the sweeps optimise, so it has to be logged."""
        self.val_acc_best(self.val_acc.compute())
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """CONTRACT 5 — build the optimiser from the config, not from a literal.

        `optimizer` and `scheduler` arrive as partials from the yaml, so the learning
        rate lives in the config and a sweep can vary it without touching this file.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.hparams.scheduler(optimizer=optimizer),
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
