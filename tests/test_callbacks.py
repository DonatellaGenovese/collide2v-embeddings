"""Tests for the callbacks, which read what a training step returns.

A callback cannot reach inside a step, so the two here depend on what the model hands
back. That coupling is easy to break by writing a new model, and it used to break
loudly: `multiclassROC` is in the default callbacks, and a model whose validation_step
returned nothing — tinyMLP, among those shipped — crashed the first validation batch
with "'NoneType' object is not subscriptable".
"""

from types import SimpleNamespace

import torch
from torchmetrics.classification import MulticlassAUROC

from src.callbacks.multiclassROC import MCROC

CLASSES = 3


def prepared_callback():
    callback = MCROC()
    callback.device = "cpu"
    callback.num_classes = CLASSES
    callback.class_names = ["alpha", "beta", "gamma"]
    callback.mcauc = MulticlassAUROC(num_classes=CLASSES, average=None)
    return callback


def a_batch():
    return torch.rand(8, 5), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])


def test_probabilities_are_accumulated():
    callback = prepared_callback()
    probs = torch.rand(8, CLASSES).softmax(dim=-1)

    callback.on_validation_batch_end(
        SimpleNamespace(num_val_batches=1), SimpleNamespace(), {"probs": probs}, a_batch(), 0
    )

    assert callback.mcauc.compute().shape == (CLASSES,)


class ModelWithoutProbs:
    """Stands in for a model whose validation_step returns None."""


def test_a_model_that_returns_nothing_does_not_break_validation(capsys):
    """The case that mattered: no probabilities, so no AUROC, but the run goes on."""
    callback = prepared_callback()
    module = ModelWithoutProbs()

    for batch_idx in range(3):
        callback.on_validation_batch_end(
            SimpleNamespace(num_val_batches=3), module, None, a_batch(), batch_idx
        )

    message = capsys.readouterr().out
    assert message.count("no 'probs'") == 1, "say it once, not once per batch"
    assert "validation_step" in message, "the message should say where to fix it"


def test_every_shipped_model_hands_over_what_the_callback_needs():
    """The contract in section 6.4, checked by reading the source rather than training.

    Training each model for a batch would need a dataset; what matters here is that none
    of them silently stops returning probs, which is the mistake this test exists for.
    """
    import inspect

    from src.models.collide2v_contrastive import COLLIDE2VContrastiveLitModule
    from src.models.collide2v_tinymlp import COLLIDE2VTinyMLPLitModule
    from src.models.collide2v_tinytransformer import COLLIDE2VTransformerLitModule
    from src.models.template_model import TemplateLitModule

    for cls in (COLLIDE2VTinyMLPLitModule, COLLIDE2VTransformerLitModule,
                TemplateLitModule, COLLIDE2VContrastiveLitModule):
        step = inspect.getsource(cls.validation_step)
        assert '"probs"' in step, f"{cls.__name__}.validation_step returns no probs"
