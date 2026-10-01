"""Tests for the check that gates training: does the dataset hold what was asked?

The check reads the shards, so what it reports is what the loader will actually
find. The cases below are the ones that used to pass or crash for the wrong
reason, when the counts came from `file_event_counts.json` instead.
"""

import os
import tempfile

import numpy as np

from src.data.utils import count_split_events, has_enough_events

CLASSNAMES = ["alpha", "beta"]
FOLDER_MAP = {"alpha": "class_alpha", "beta": "class_beta"}
SPLITS = [20, 10, 10]


def write_dataset(base_dir, rows=None):
    """train/val/test per class, each split made of two shards.

    `rows` overrides the rows of one shard, as (split, class, shard index, rows),
    which is how a short shard is simulated.
    """
    override = {} if rows is None else {rows[:3]: rows[3]}
    for split, per_shard in zip(("train", "val", "test"), (15, 8, 8)):
        for class_index, cname in enumerate(CLASSNAMES):
            split_dir = os.path.join(base_dir, split, FOLDER_MAP[cname])
            os.makedirs(split_dir)
            for shard in range(2):
                n = override.get((split, cname, shard), per_shard)
                np.save(os.path.join(split_dir, f"shard{shard}_x.npy"),
                        np.zeros((n, 3), dtype=np.float32))
                np.save(os.path.join(split_dir, f"shard{shard}_y.npy"),
                        np.full(n, class_index, dtype=np.int64))


def test_counts_come_from_the_shards():
    with tempfile.TemporaryDirectory() as tmp:
        write_dataset(tmp)
        events, shards = count_split_events(os.path.join(tmp, "train", "class_alpha"))

        assert (events, shards) == (30, 2)


def test_a_complete_dataset_passes():
    with tempfile.TemporaryDirectory() as tmp:
        write_dataset(tmp)

        assert has_enough_events(tmp, SPLITS, CLASSNAMES, FOLDER_MAP) is True


def test_a_short_shard_is_noticed():
    """The case the snapshot could not see: the file is there, the events are not.

    Reading counts from `file_event_counts.json` described the Parquet file, so a
    shard written short — a job killed while writing, a partial copy — counted as
    full and training started on less data than it asked for.
    """
    with tempfile.TemporaryDirectory() as tmp:
        write_dataset(tmp, rows=("train", "alpha", 1, 1))  # 15 + 1 events, 20 needed

        assert has_enough_events(tmp, SPLITS, CLASSNAMES, FOLDER_MAP) is False


def test_a_missing_split_directory_fails():
    with tempfile.TemporaryDirectory() as tmp:
        write_dataset(tmp)
        for f in os.listdir(os.path.join(tmp, "test", "class_beta")):
            os.remove(os.path.join(tmp, "test", "class_beta", f))
        os.rmdir(os.path.join(tmp, "test", "class_beta"))

        assert has_enough_events(tmp, SPLITS, CLASSNAMES, FOLDER_MAP) is False


def test_an_empty_split_directory_fails():
    with tempfile.TemporaryDirectory() as tmp:
        write_dataset(tmp)
        split_dir = os.path.join(tmp, "val", "class_alpha")
        for f in os.listdir(split_dir):
            os.remove(os.path.join(split_dir, f))

        assert has_enough_events(tmp, SPLITS, CLASSNAMES, FOLDER_MAP) is False


def test_a_class_outside_the_snapshot_is_fine():
    """Shards whose names no event-count snapshot knows: the old check raised KeyError.

    This is the normal case for data downloaded from Hugging Face, and for any
    process added to the dataset after the snapshot was taken.
    """
    with tempfile.TemporaryDirectory() as tmp:
        write_dataset(tmp)
        for split in ("train", "val", "test"):
            split_dir = os.path.join(tmp, split, FOLDER_MAP["alpha"])
            for f in sorted(os.listdir(split_dir)):
                os.rename(os.path.join(split_dir, f), os.path.join(split_dir, f"unknown_{f}"))

        assert has_enough_events(tmp, SPLITS, CLASSNAMES, FOLDER_MAP) is True


def test_a_missing_target_fails_without_raising():
    assert has_enough_events("/does/not/exist", SPLITS, CLASSNAMES, FOLDER_MAP) is False
    assert has_enough_events("", SPLITS, CLASSNAMES, FOLDER_MAP) is False
