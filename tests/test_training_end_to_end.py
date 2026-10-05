"""Train every model for a few batches, through the real entry point, on fake data.

The other tests check the pieces: the manifest, the loader, the loss, a model built
against a stand-in datamodule. None of them composed the real configs with Hydra and
handed them to `src/train.py`, and two failures lived exactly in that gap, found only by
running training on lxplus:

  - `configs/model/scheduler/none.yaml` reached the models as an empty mapping instead of
    None, so every model left at the default scheduler crashed in configure_optimizers;
  - the datamodule set the event width in prepare_data(), which returns early when the
    dataset exists, so models were built for the raw width and failed on the first batch.

So this builds a small dataset with the real pipeline, then calls `train()` and the probe
on it for each model, with the configs as a user would compose them.
"""

import os

import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from tests.test_pipeline_determinism import (
    CLASSES,
    FOLDERS,
    SPLITS,
    run_pipeline,
    write_fake_dataset,
)

LABEL = "fake_two_class"
MODELS = ["tinyMLP", "tinyTransformer", "template", "simclr", "supcon"]


@pytest.fixture(scope="module")
def prepared_dataset(tmp_path_factory):
    """A preprocessed two-class dataset, laid out where the data config expects it."""
    root = tmp_path_factory.mktemp("training")
    dataset_dir = root / "parquet"
    filelist = write_fake_dataset(str(dataset_dir))
    run_pipeline(str(root / "data" / LABEL), str(dataset_dir), filelist)
    return root


def composed(root, model, extra=()):
    """The training config as `python src/train.py ...` would compose it."""
    GlobalHydra.instance().clear()
    overrides = [
        "experiment=fm_testing_binary",
        f"model={model}",
        "trainer=cpu",
        "logger=csv",
        f"paths.eos_data_dir={root}/data",
        f"paths.tmp_data_dir={root}/tmp",
        f"paths.log_dir={root}/logs",
        f"data.label={LABEL}",
        f"data.to_classify=[{','.join(CLASSES)}]",
        *[f"+data.process_to_folder.{c}={FOLDERS[c]}" for c in CLASSES],
        f"data.train_val_test_split_per_class=[{','.join(map(str, SPLITS))}]",
        "data.batch_size=8",
        "data.num_workers=0",
        "trainer.max_epochs=1",
        "+trainer.limit_train_batches=3",
        "+trainer.limit_val_batches=2",
        "+trainer.limit_test_batches=2",
        *extra,
    ]
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=overrides)
    with open_dict(cfg):
        cfg.paths.root_dir = str(root)
        cfg.paths.output_dir = str(root / "runs" / model)
        cfg.hydra.runtime.output_dir = cfg.paths.output_dir
        cfg.hydra.job.num = 0
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
        # The small experiment's real model sizes, but a CPU should finish in seconds.
        if "d_model" in cfg.model:
            cfg.model.d_model, cfg.model.n_heads = 16, 2
            cfg.model.num_layers, cfg.model.d_ff = 1, 32
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    return cfg


@pytest.mark.parametrize("model", MODELS)
def test_every_model_trains_and_tests_through_train_py(prepared_dataset, model):
    from src.train import train

    cfg = composed(prepared_dataset, model)
    HydraConfig().set_config(cfg)

    metric_dict, _ = train(cfg)

    assert any(k.startswith("test/") for k in metric_dict), f"no test metrics: {sorted(metric_dict)}"
    checkpoints = os.path.join(cfg.paths.output_dir, "checkpoints")
    assert os.path.exists(os.path.join(checkpoints, "last.ckpt")), "no checkpoint written"


def test_a_contrastive_run_without_its_head_trains_as_documented(prepared_dataset):
    """The recipe in docs/logging.md: no classification head, so no class probabilities.

    Validation and test then return nothing for the ROC callback, which has to step
    aside at the end of each epoch as well as on each batch.
    """
    from src.train import train

    cfg = composed(prepared_dataset, "supcon", extra=[
        "model.use_classification_head=false",
        "callbacks.model_checkpoint.monitor=val/contrastive_loss",
        "callbacks.model_checkpoint.mode=min",
        "callbacks.early_stopping.monitor=val/contrastive_loss",
        "callbacks.early_stopping.mode=min",
    ])
    with open_dict(cfg):
        cfg.paths.output_dir = str(prepared_dataset / "runs" / "supcon_no_head")
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    HydraConfig().set_config(cfg)

    metric_dict, _ = train(cfg)

    assert "test/contrastive_loss" in metric_dict


@pytest.mark.parametrize("model", ["tinyMLP", "simclr"])
def test_the_probe_runs_on_a_checkpoint_from_train_py(prepared_dataset, model):
    """eval_probes.py, end to end, on what the test above just wrote."""
    from src.eval_probes import main as probe_main

    ckpt = prepared_dataset / "runs" / model / "checkpoints" / "last.ckpt"
    if not ckpt.exists():
        pytest.skip("the training test for this model has to run first")

    GlobalHydra.instance().clear()
    cfg = composed(prepared_dataset, model)  # same data and sizes as the run
    with open_dict(cfg):
        cfg.ckpt_path = str(ckpt)
        cfg.paths.output_dir = str(prepared_dataset / "runs" / f"probe_{model}")
        cfg.probe = {"train_events": 80, "epochs": 5, "batch_size": 16}
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    # A probe is run as its own command, in a fresh interpreter. Training in this process
    # loaded a checkpoint for its test pass, and Lightning allowlisted functools.partial
    # for torch.load while doing it — which once hid a probe that failed everywhere else.
    torch.serialization.clear_safe_globals()

    probe_main.__wrapped__(cfg)

    assert os.path.exists(os.path.join(cfg.paths.output_dir, "probe_results.json"))


def test_the_width_a_model_is_built_for_is_the_preprocessed_one(prepared_dataset):
    """And not the raw width, even when prepare_data() returns early."""
    import hydra

    cfg = composed(prepared_dataset, "tinyMLP")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()  # the dataset exists, so this returns at once
    datamodule.setup("fit")

    x, _ = next(iter(datamodule.train_dataloader()))
    assert datamodule.vlen == x.shape[1], (
        f"models would be built for {datamodule.vlen} features and receive {x.shape[1]}"
    )


def test_no_model_receives_an_uncallable_scheduler():
    """`scheduler: none` has to arrive as None, for every model config."""
    import hydra

    for model in MODELS:
        GlobalHydra.instance().clear()
        with initialize(version_base="1.3", config_path="../configs"):
            cfg = compose(config_name="train.yaml",
                          overrides=["experiment=fm_testing_binary", f"model={model}"])
        scheduler = hydra.utils.instantiate(cfg.model).hparams.scheduler
        assert scheduler is None or callable(scheduler), (
            f"{model} receives scheduler={scheduler!r}, which configure_optimizers would call"
        )

