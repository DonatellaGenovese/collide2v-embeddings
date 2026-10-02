"""Is the dataset the default config points at actually on this machine?

The tests inherited from the lightning-hydra template call `train()` on the default
configuration, so they need a preprocessed COLLIDE-2V dataset. On a laptop, or on a
batch node without it, they used to start vectorising the full dataset; now they get
the refusal from `prepare_data` instead. Either way they cannot pass, so they are
skipped, and the reason says what is missing rather than failing a run that was never
going to work.
"""

import os

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


def preprocessed_dataset_dir() -> str:
    """Where the default configuration expects its preprocessed shards, or ""."""
    try:
        GlobalHydra.instance().clear()
        with initialize(version_base="1.3", config_path="../../configs"):
            cfg = compose(config_name="train.yaml", overrides=[])
        return str(cfg.data.paths.eos_preproc_dir)
    except Exception:
        return ""
    finally:
        GlobalHydra.instance().clear()


def dataset_is_available() -> bool:
    """True when that directory exists and holds at least one class directory."""
    target = preprocessed_dataset_dir()
    if not target or not os.path.isdir(target):
        return False
    for split in ("train", "val", "test"):
        split_dir = os.path.join(target, split)
        if not os.path.isdir(split_dir) or not os.listdir(split_dir):
            return False
    return True


SKIP_REASON = (
    "needs a preprocessed COLLIDE-2V dataset at the path the default config points at; "
    "produce one with `python src/prepare_data.py experiment=<name>`"
)
