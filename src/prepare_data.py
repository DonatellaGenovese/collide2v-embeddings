"""Produce the vectorised and preprocessed shards that a training run reads.

This is the one command that writes a dataset. Training does not: `src/train.py`
stops if the data it needs is missing, because vectorising the full dataset takes
days and a training job would spend its GPU allocation doing it.

Use it for a dataset small enough to build on one machine — a few files per class,
which is the normal case off lxplus. For the full dataset, send the work to the
batch system instead:

    python scripts/submit_vectorization_jobs.py experiment=<name>   # wait for these
    python scripts/submit_preprocessing_jobs.py experiment=<name>

Both routes write the same files, in the same layout, from the same manifest.

Usage:
    python src/prepare_data.py experiment=fm_testing_18class_highlevel
    python src/prepare_data.py experiment=<name> data.label=my_small_test
    python src/prepare_data.py experiment=<name> preprocess.enabled=false

Rerunning is safe: files already produced are skipped, and a dataset is never
rebuilt with a different file selection — see section 5 of the README.
"""

import hydra
import rootutils
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="vectorize_preprocess.yaml")
def main(cfg: DictConfig) -> None:
    """Vectorise, then preprocess, the dataset the config describes."""
    with open_dict(cfg):
        # The flag exists so that training cannot do this by accident; here it is
        # exactly what was asked for.
        cfg.data.allow_data_preparation = True

    log.info(f"Dataset label: {cfg.data.label}")
    log.info(f"Reading Parquet from: {cfg.paths.dataset_dir}")
    log.info(f"Writing shards to:   {cfg.data.paths.eos_vec_dir}")
    log.info(f"Classes: {list(cfg.data.to_classify)}")
    log.info(f"Events per class [train, val, test]: {list(cfg.data.train_val_test_split_per_class)}")
    log.info(f"Seed {cfg.data.seed}, manifest strategy {cfg.data.manifest_strategy}")

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()

    log.info("Done. Train with the same experiment, which will find this dataset in place.")


if __name__ == "__main__":
    main()
