"""Tests for the contrastive objective, its loss and its augmentations.

The loss here replaces two near-identical implementations of about a thousand lines
each, which differed in 68 lines — the published `collide2v_augmented_supcon.py` and
`collide2v_augmented_selfsupcon.py`. The values pinned below were produced by running
those two on the inputs this file builds, so the rewrite is checked against them rather
than trusted.
"""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.data.augmentations import PhysicsAugmentation
from src.models.collide2v_contrastive import COLLIDE2VContrastiveLitModule
from src.models.components.contrastive import (
    ContrastiveLoss,
    ProjectionHead,
    RandomMaskingAugmentation,
)

BATCH, VIEWS, DIM, CLASSES = 16, 2, 8, 4

# Produced by the originals on the inputs of reference_batch(), temperature and
# base_temperature both 0.07, which is what the published runs used.
SUPCON_REFERENCE = 9.3070850372
SIMCLR_REFERENCE = 8.7154512405

FEATURE_MAP = {
    "jets": {
        "start": 0,
        "end": 24,
        "topk": 4,
        "columns": ["FullReco_JetPuppiAK4_PT", "FullReco_JetPuppiAK4_Eta",
                    "FullReco_JetPuppiAK4_Phi_sin", "FullReco_JetPuppiAK4_Phi_cos",
                    "FullReco_JetPuppiAK4_Mass", "FullReco_JetPuppiAK4_BTag"],
        "count": False,
    },
    "puppi_met": {
        "start": 24,
        "end": 26,
        "topk": None,
        "columns": ["FullReco_PUPPIMET_MET", "FullReco_PUPPIMET_Phi_sin"],
        "count": False,
    },
}
WIDTH = 26


def reference_batch():
    """The exact inputs the pinned numbers were computed on."""
    torch.manual_seed(0)
    features = F.normalize(torch.randn(BATCH * VIEWS, DIM), dim=-1)
    labels = torch.randint(0, CLASSES, (BATCH,))
    return features, labels


# ---------------------------------------------------------------------------
# The loss, against the implementations it replaces
# ---------------------------------------------------------------------------


def test_supcon_matches_the_published_implementation():
    features, labels = reference_batch()

    loss = ContrastiveLoss(temperature=0.07, base_temperature=0.07)(
        features, labels.repeat(VIEWS), require_positives=False
    )

    assert loss.item() == pytest.approx(SUPCON_REFERENCE, abs=1e-6)


def test_simclr_matches_the_published_implementation():
    features, _ = reference_batch()
    instance_ids = torch.arange(BATCH).repeat(VIEWS)

    loss = ContrastiveLoss(temperature=0.07, base_temperature=0.07)(
        features, instance_ids, require_positives=False
    )

    assert loss.item() == pytest.approx(SIMCLR_REFERENCE, abs=1e-6)


def test_the_two_regimes_are_different_losses():
    """Same features, different notion of a positive, so different numbers."""
    features, labels = reference_batch()
    loss = ContrastiveLoss(temperature=0.07, base_temperature=0.07)

    by_label = loss(features, labels.repeat(VIEWS), require_positives=False)
    by_view = loss(features, torch.arange(BATCH).repeat(VIEWS), require_positives=False)

    assert not torch.isclose(by_label, by_view)


def test_identical_views_give_a_lower_loss_than_random_ones():
    """The loss has to reward what it is supposed to reward."""
    torch.manual_seed(1)
    half = F.normalize(torch.randn(BATCH, DIM), dim=-1)
    loss = ContrastiveLoss(temperature=0.07, base_temperature=0.07)
    ids = torch.arange(BATCH).repeat(2)

    aligned = loss(torch.cat([half, half]), ids, require_positives=False)
    scrambled = loss(torch.cat([half, F.normalize(torch.randn(BATCH, DIM), dim=-1)]), ids,
                     require_positives=False)

    assert aligned < scrambled


def test_a_single_group_is_refused_while_training():
    features = F.normalize(torch.randn(8, DIM), dim=-1)
    one_group = torch.zeros(8, dtype=torch.long)
    loss = ContrastiveLoss()

    with pytest.raises(ValueError) as err:
        loss(features, one_group, require_positives=True)
    assert "single-class batch" in str(err.value)

    # Off, it is zero and still differentiable, so a degenerate validation batch does
    # not stop a run.
    quiet = loss(features, one_group, require_positives=False)
    assert quiet.item() == 0.0
    assert quiet.requires_grad


def test_mismatched_group_ids_are_refused():
    features = F.normalize(torch.randn(8, DIM), dim=-1)

    with pytest.raises(ValueError):
        ContrastiveLoss()(features, torch.arange(4), require_positives=False)


def test_the_projection_head_changes_the_dimension_only():
    head = ProjectionHead(input_dim=DIM, hidden_dim=16, output_dim=4)

    out = head(torch.randn(BATCH, DIM))

    assert out.shape == (BATCH, 4)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def test_masking_zeroes_roughly_the_requested_fraction():
    torch.manual_seed(0)
    x = torch.rand(256, WIDTH) + 1.0  # no zeros, so masked entries are visible
    untouched = x.clone()
    augment = RandomMaskingAugmentation(FEATURE_MAP, mask_probability=0.25)

    out = augment(x)
    fraction = (out == 0).float().mean().item()

    assert 0.2 < fraction < 0.3
    assert torch.equal(x, untouched), "the input must not be modified in place"


def test_masking_whole_objects_removes_them_entirely():
    torch.manual_seed(0)
    x = torch.rand(64, WIDTH) + 1.0
    augment = RandomMaskingAugmentation(FEATURE_MAP, mask_probability=0.5,
                                        mask_full_particle=True)

    out = augment(x)

    # Every jet slot is either untouched or zero across all six of its columns.
    for k in range(FEATURE_MAP["jets"]["topk"]):
        block = out[:, k * 6:(k + 1) * 6]
        zeros_per_object = (block == 0).sum(dim=1)
        assert set(zeros_per_object.unique().tolist()) <= {0, 6}
    # MET is a scalar group and is never masked this way.
    assert (out[:, 24:26] != 0).all()


def write_preprocessing_metadata(directory):
    with open(os.path.join(directory, "feature_map.json"), "w") as f:
        json.dump(FEATURE_MAP, f)
    with open(os.path.join(directory, "norm_stats.json"), "w") as f:
        json.dump({"block_group_stats": {"jets.eta": {"iqr": 2.0}, "jets.pt": {"iqr": 4.0},
                                         "puppi_met.met": {"iqr": 3.0}}}, f)


def test_phi_rotation_keeps_the_pair_on_the_unit_circle():
    """A rotation is only valid if it leaves sin^2 + cos^2 alone."""
    with tempfile.TemporaryDirectory() as tmp:
        write_preprocessing_metadata(tmp)
        augment = PhysicsAugmentation(
            os.path.join(tmp, "feature_map.json"), os.path.join(tmp, "norm_stats.json"),
            phi_rotation=True, eta_boost=False, pt_smearing=False,
        )
        torch.manual_seed(0)
        phi = torch.rand(32) * 6.28
        x = torch.zeros(32, WIDTH)
        for k in range(4):
            x[:, k * 6 + 2] = torch.sin(phi)
            x[:, k * 6 + 3] = torch.cos(phi)

        out = augment(x)

        for k in range(4):
            radius = out[:, k * 6 + 2] ** 2 + out[:, k * 6 + 3] ** 2
            assert torch.allclose(radius, torch.ones_like(radius), atol=1e-5)
        # The same rotation for every object of an event, or the event is deformed.
        first = torch.atan2(out[:, 2], out[:, 3]) - phi
        second = torch.atan2(out[:, 8], out[:, 9]) - phi
        assert torch.allclose(torch.remainder(first - second, 6.283185), torch.zeros(32), atol=1e-4)


def test_the_eta_boost_is_one_shift_per_event_and_leaves_padding_alone():
    with tempfile.TemporaryDirectory() as tmp:
        write_preprocessing_metadata(tmp)
        augment = PhysicsAugmentation(
            os.path.join(tmp, "feature_map.json"), os.path.join(tmp, "norm_stats.json"),
            phi_rotation=False, eta_boost=True, pt_smearing=False, eta_sigma=1.5,
        )
        x = torch.zeros(64, WIDTH)
        x[:, 1] = 0.5          # first jet has an eta
        x[:, 7] = -0.5         # second jet too
        # third and fourth jets are empty, as padding

        out = augment(x)

        shift_first = out[:, 1] - 0.5
        shift_second = out[:, 7] + 0.5
        assert torch.allclose(shift_first, shift_second, atol=1e-6), "one boost per event"
        assert (out[:, 13] == 0).all() and (out[:, 19] == 0).all(), "padding must stay padding"
        assert shift_first.abs().mean() > 0


def test_pt_smearing_is_independent_per_object():
    with tempfile.TemporaryDirectory() as tmp:
        write_preprocessing_metadata(tmp)
        augment = PhysicsAugmentation(
            os.path.join(tmp, "feature_map.json"), os.path.join(tmp, "norm_stats.json"),
            phi_rotation=False, eta_boost=False, pt_smearing=True, pt_sigma=0.2,
        )
        x = torch.zeros(256, WIDTH)
        x[:, 0] = 1.0
        x[:, 6] = 1.0

        out = augment(x)

        first, second = out[:, 0] - 1.0, out[:, 6] - 1.0
        assert not torch.allclose(first, second, atol=1e-3), "resolution is per object"
        # sigma / iqr, with iqr = 4.0 for the jet pt block
        assert 0.02 < first.std().item() < 0.08


# ---------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------


def build_module(positives, tmp, **kwargs):
    """The module as Lightning would build it, with a stand-in for the datamodule.

    `setup()` reads the feature map and the number of classes off the datamodule, which
    is the contract every model here follows, so a test has to provide one.
    """
    write_preprocessing_metadata(tmp)
    module = COLLIDE2VContrastiveLitModule(
        positives=positives, d_model=16, n_heads=2, num_layers=1, d_ff=32,
        projection_dim=8, hidden_projection_dim=16, num_views=2,
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1), **kwargs
    )
    module._trainer = SimpleNamespace(
        datamodule=SimpleNamespace(paths={"eos_preproc_dir": tmp}, num_classes=CLASSES)
    )
    module.setup("fit")
    return module


@pytest.mark.parametrize("positives", ["views", "labels"])
def test_a_training_step_runs_and_produces_gradients(positives):
    with tempfile.TemporaryDirectory() as tmp:
        module = build_module(positives, tmp)
        module.train()
        # The stand-in datamodule was only needed to build the model. Lightning's
        # self.log() tolerates a module with no trainer — it warns and returns — which
        # is what lets a training step be exercised without starting a Trainer.
        module._trainer = None
        batch = (torch.rand(8, WIDTH), torch.arange(8) % CLASSES)

        loss = module.training_step(batch, batch_idx=0)
        loss.backward()

        assert torch.isfinite(loss)
        grads = [p.grad for p in module.encoder.parameters() if p.grad is not None]
        assert grads, "the encoder received no gradient"
        assert any(g.abs().sum() > 0 for g in grads)


def test_the_two_regimes_group_the_batch_differently():
    with tempfile.TemporaryDirectory() as tmp:
        labels = torch.tensor([0, 0, 1, 1])

        by_view = build_module("views", tmp).group_ids(labels, batch_size=4, n_views=2)
        by_label = build_module("labels", tmp).group_ids(labels, batch_size=4, n_views=2)

        assert by_view.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
        assert by_label.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


def test_embeddings_come_from_the_encoder_not_the_projection():
    with tempfile.TemporaryDirectory() as tmp:
        module = build_module("labels", tmp)
        module.eval()
        x = torch.rand(4, WIDTH)

        embeddings, projections, logits = module(x)

        assert embeddings.shape == (4, 16), "d_model wide, the thing that is kept"
        assert projections.shape == (4, 8), "projection_dim wide, thrown away after training"
        assert torch.allclose(projections.norm(dim=-1), torch.ones(4), atol=1e-5)
        assert logits.shape == (4, CLASSES)
        assert torch.equal(module.get_embeddings(x), embeddings)


def test_simclr_refuses_a_single_view():
    with pytest.raises(ValueError) as err:
        COLLIDE2VContrastiveLitModule(positives="views", num_views=1)
    assert "num_views" in str(err.value)


def test_an_unknown_regime_is_refused():
    with pytest.raises(ValueError):
        COLLIDE2VContrastiveLitModule(positives="class_labels")


# ---------------------------------------------------------------------------
# The linear probe
# ---------------------------------------------------------------------------


def test_a_probe_recovers_a_structure_that_is_there():
    """On embeddings that separate the classes, a linear probe should do well.

    The point of the probe is to measure the representation, so the test fixes the
    representation and checks the measurement: four well-separated clusters, one per
    class, must be nearly perfectly classified by a linear map.
    """
    from src.eval_probes import score, train_linear_probe

    torch.manual_seed(0)
    centres = torch.eye(CLASSES, 6) * 5.0

    def sample(n):
        labels = torch.randint(0, CLASSES, (n,))
        return centres[labels] + torch.randn(n, 6), labels

    probe, _ = train_linear_probe(sample(800), sample(200), CLASSES, epochs=30,
                                  batch_size=128)
    results = score(probe, sample(200), CLASSES)

    assert results["accuracy"] > 0.95
    assert results["auroc"] > 0.99


def test_a_probe_finds_nothing_in_an_embedding_that_holds_nothing():
    """And on noise it must report chance, or it is measuring itself."""
    from src.eval_probes import score, train_linear_probe

    torch.manual_seed(0)

    def noise(n):
        return torch.randn(n, 6), torch.randint(0, CLASSES, (n,))

    probe, _ = train_linear_probe(noise(800), noise(200), CLASSES, epochs=30,
                                  batch_size=128)
    results = score(probe, noise(400), CLASSES)

    assert results["accuracy"] < 0.45, "a linear probe cannot classify noise"


# ---------------------------------------------------------------------------
# The template, and the contract it demonstrates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_path,kwargs", [
    ("src.models.template_model.TemplateLitModule", dict(hidden_dim=16)),
    ("src.models.collide2v_contrastive.COLLIDE2VContrastiveLitModule",
     dict(d_model=16, n_heads=2, num_layers=1, d_ff=32, projection_dim=8,
          hidden_projection_dim=16)),
])
def test_a_model_builds_from_the_datamodule_and_exposes_embeddings(module_path, kwargs, tmp_path):
    """The contract the README documents, checked on the template and on a real model.

    A model takes hyperparameters only, builds its layers in setup() from the
    datamodule, and offers get_embeddings(). If this test fails, either the contract
    moved or the template stopped being an example of it.
    """
    import importlib

    write_preprocessing_metadata(str(tmp_path))
    module_name, class_name = module_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)

    model = cls(optimizer=lambda params: torch.optim.SGD(params, lr=0.1), **kwargs)
    assert model.training_step is not None

    model._trainer = SimpleNamespace(
        datamodule=SimpleNamespace(vlen=WIDTH, num_classes=CLASSES,
                                   paths={"eos_preproc_dir": str(tmp_path)})
    )
    model.setup("fit")
    model.setup("validate")  # must be idempotent

    x = torch.rand(4, WIDTH)
    embeddings = model.get_embeddings(x)
    assert embeddings.ndim == 2 and embeddings.shape[0] == 4

    model._trainer = None
    model.train()
    loss = model.training_step((x, torch.arange(4) % CLASSES), batch_idx=0)
    loss.backward()
    assert torch.isfinite(loss)


def test_the_template_refuses_to_build_without_a_datamodule():
    from src.models.template_model import TemplateLitModule

    with pytest.raises(RuntimeError) as err:
        TemplateLitModule().setup("fit")
    assert "datamodule" in str(err.value)
