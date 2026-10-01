"""Tests for the loader: which events an epoch contains, and how many.

The property that matters is that the sample is a property of the dataset and the
seed, not of the machine it runs on: the same events, in the same number, whether
the DataLoader uses no workers or several.
"""

import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.datasets import LocalVectorDataset

CLASSNAMES = ["alpha", "beta"]
FOLDER_MAP = {"alpha": "class_alpha", "beta": "class_beta"}


def write_shards(base_dir, rows_per_shard=(10, 10, 10, 10)):
    """One directory per class, with shards whose every row is identifiable.

    Feature 0 of a row is `class_index * 1e6 + shard_index * 1e3 + row`, so a
    sample can be checked for duplicates and for which rows of which shard it
    took.
    """
    for class_index, cname in enumerate(CLASSNAMES):
        class_dir = os.path.join(base_dir, FOLDER_MAP[cname])
        os.makedirs(class_dir)
        for shard_index, rows in enumerate(rows_per_shard):
            ids = np.array(
                [class_index * 1_000_000 + shard_index * 1_000 + row for row in range(rows)],
                dtype=np.float32,
            )
            X = np.stack([ids, np.arange(rows, dtype=np.float32)], axis=1)
            y = np.full(rows, class_index, dtype=np.int64)
            np.save(os.path.join(class_dir, f"shard{shard_index:02d}_x.npy"), X)
            np.save(os.path.join(class_dir, f"shard{shard_index:02d}_y.npy"), y)


def dataset(base_dir, per_class_limit, seed=1, shuffle=False):
    return LocalVectorDataset(
        base_dir,
        per_class_limit=per_class_limit,
        shuffle_file_order=shuffle,
        classnames=CLASSNAMES,
        folder_map=FOLDER_MAP,
        seed=seed,
    )


def ids_and_labels(loader_or_dataset):
    ids, labels = [], []
    for x, y in loader_or_dataset:
        ids.extend(np.atleast_1d(x[..., 0].numpy().ravel()).tolist())
        labels.extend(np.atleast_1d(y.numpy().ravel()).tolist())
    return ids, labels


def test_exactly_the_requested_events_per_class():
    with tempfile.TemporaryDirectory() as tmp:
        write_shards(tmp)  # 40 events per class
        ds = dataset(tmp, per_class_limit=25)

        ids, labels = ids_and_labels(ds)

        assert ds.num_events == 50, "num_events must be the events of one epoch"
        assert len(ids) == 50
        assert len(set(ids)) == 50, "no event may be yielded twice"
        assert labels.count(0) == 25 and labels.count(1) == 25


@pytest.mark.parametrize("num_workers", [0, 1, 3, 5])
def test_sample_is_the_same_whatever_the_number_of_workers(num_workers):
    with tempfile.TemporaryDirectory() as tmp:
        write_shards(tmp)
        ds = dataset(tmp, per_class_limit=25)
        expected = sorted(ids_and_labels(dataset(tmp, per_class_limit=25))[0])

        loader = DataLoader(ds, batch_size=7, num_workers=num_workers)
        ids, labels = ids_and_labels(loader)

        assert sorted(ids) == expected
        assert labels.count(0) == 25 and labels.count(1) == 25


def test_a_different_seed_takes_different_events():
    with tempfile.TemporaryDirectory() as tmp:
        write_shards(tmp)

        first = sorted(ids_and_labels(dataset(tmp, per_class_limit=25, seed=1))[0])
        again = sorted(ids_and_labels(dataset(tmp, per_class_limit=25, seed=1))[0])
        other = sorted(ids_and_labels(dataset(tmp, per_class_limit=25, seed=2))[0])

        assert first == again
        assert first != other


def test_no_limit_reads_every_event():
    with tempfile.TemporaryDirectory() as tmp:
        write_shards(tmp)
        ds = dataset(tmp, per_class_limit=None)

        ids, labels = ids_and_labels(ds)

        assert ds.num_events == 80
        assert len(set(ids)) == 80
        assert labels.count(0) == 40 and labels.count(1) == 40


def test_rows_of_a_partially_used_shard_are_not_taken_from_the_top():
    """The old loader always took the first rows of a file; this one draws them."""
    with tempfile.TemporaryDirectory() as tmp:
        write_shards(tmp, rows_per_shard=(100,))  # one shard, so it must be split
        ds = dataset(tmp, per_class_limit=30)

        ids, _ = ids_and_labels(ds)
        rows_of_alpha = sorted(int(i) % 1_000 for i in ids if i < 1_000_000)

        assert len(rows_of_alpha) == 30
        assert rows_of_alpha != list(range(30))
        assert max(rows_of_alpha) >= 30, "the draw should reach beyond the first rows"


def test_asking_for_more_events_than_exist_takes_all_of_them():
    with tempfile.TemporaryDirectory() as tmp:
        write_shards(tmp)
        ds = dataset(tmp, per_class_limit=1000)

        ids, labels = ids_and_labels(ds)

        assert ds.num_events == 80
        assert len(set(ids)) == 80
        assert labels.count(0) == 40 and labels.count(1) == 40


def test_shuffling_mixes_the_classes_without_changing_the_sample():
    with tempfile.TemporaryDirectory() as tmp:
        write_shards(tmp)

        ordered, ordered_labels = ids_and_labels(dataset(tmp, per_class_limit=25, shuffle=False))
        shuffled, shuffled_labels = ids_and_labels(dataset(tmp, per_class_limit=25, shuffle=True))

        assert sorted(ordered) == sorted(shuffled), "shuffling must not change which events"
        assert ordered_labels != shuffled_labels, "shuffling should interleave the classes"
        # Without shuffling the first shard read belongs to the first class.
        assert ordered_labels[0] == 0
