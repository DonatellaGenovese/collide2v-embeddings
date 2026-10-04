"""Tests for the ways a long run used to fail late, quietly, or slowly.

Each of these was a known limitation in section 4.6 of the README: a Parquet file that
could not be read was skipped while every job reported success; the default output
directory could fill an AFS quota halfway through; and checking a dataset opened every
shard, which on EOS took a minute and a half for twelve classes.
"""

import ast
import json
import os
import pathlib
import time

import numpy as np
import pytest

from src.data.utils import (
    SHARD_ROWS_CACHE,
    check_output_space,
    compute_vlen,
    count_split_events,
    get_all_cols,
    make_split_manifest,
    parse_fs_listquota,
    shard_rows,
    vectorize_to_local,
)
from tests.test_pipeline_determinism import (
    CLASSES,
    DATASETS_CONFIG,
    FOLDERS,
    SPLITS,
    write_fake_dataset,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# An unreadable Parquet file stops the run, after the others are done
# ---------------------------------------------------------------------------


def vectorize(dataset_dir, out, manifest, **kwargs):
    vectorize_to_local(
        base_dir=dataset_dir,
        config=DATASETS_CONFIG,
        class_names=CLASSES,
        folder_map=FOLDERS,
        labels_map={c: i for i, c in enumerate(CLASSES)},
        all_cols=get_all_cols(DATASETS_CONFIG),
        vlen=compute_vlen(DATASETS_CONFIG),
        tmp_vec_dir=os.path.join(out, "tmp"),
        eos_vec_dir=os.path.join(out, "vectorized"),
        split_counts=SPLITS,
        split_manifest=manifest,
        **kwargs,
    )


def corrupt_one_file(tmp_path):
    dataset = tmp_path / "dataset"
    filelist = write_fake_dataset(str(dataset))
    manifest = make_split_manifest(filelist, SPLITS, list(filelist), seed=3)
    broken_folder = FOLDERS[CLASSES[0]]
    broken_name = manifest[broken_folder]["val"][0]
    (dataset / broken_folder / broken_name).write_bytes(b"this is not a parquet file")
    return str(dataset), manifest, broken_folder, broken_name


def written_shards(out):
    return sorted(str(p.relative_to(out)) for p in pathlib.Path(out).rglob("*_x.npy"))


def test_an_unreadable_file_stops_the_run_and_is_named(tmp_path):
    dataset, manifest, folder, broken = corrupt_one_file(tmp_path)

    with pytest.raises(RuntimeError) as err:
        vectorize(dataset, str(tmp_path / "out"), manifest)

    assert broken in str(err.value), "the error has to say which file"
    assert "skip_unreadable_files" in str(err.value), "and what to do about it"


def test_the_other_files_are_still_written_before_it_stops(tmp_path):
    """One bad file must not throw away the rest of a batch job's work."""
    dataset, manifest, folder, broken = corrupt_one_file(tmp_path)
    out = tmp_path / "out"

    with pytest.raises(RuntimeError):
        vectorize(dataset, str(out), manifest)

    every_file = sum(len(files) for buckets in manifest.values() for files in buckets.values())
    assert len(written_shards(out)) == every_file - 1


def test_skipping_is_a_choice_that_has_to_be_made(tmp_path, capsys):
    dataset, manifest, folder, broken = corrupt_one_file(tmp_path)

    vectorize(dataset, str(tmp_path / "out"), manifest, skip_unreadable_files=True)

    assert broken in capsys.readouterr().out, "skipped, but still reported"


def test_every_option_that_changes_the_data_reaches_the_batch_job():
    """The batch route and the local route must build the same dataset from one config.

    `drop_empty_events` was not passed by src/data/vectorize_job.py, so batch jobs kept
    empty events whatever the config said, while src/prepare_data.py dropped them. This
    reads the keyword arguments of the vectorize_to_local call in both places.
    """
    must_pass = {"drop_empty_events", "skip_unreadable_files"}

    def keywords_of_call(path):
        tree = ast.parse((REPO / path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "vectorize_to_local":
                return {kw.arg for kw in node.keywords}
        raise AssertionError(f"no call to vectorize_to_local in {path}")

    for path in ("src/data/vectorize_job.py", "src/data/collide2v_datamodule.py"):
        missing = must_pass - keywords_of_call(path)
        assert not missing, f"{path} does not pass {sorted(missing)}"


# ---------------------------------------------------------------------------
# Will it fit?
# ---------------------------------------------------------------------------


def test_the_afs_quota_is_read_from_fs_listquota():
    output = """Volume Name                    Quota       Used %Used   Partition
user.dgenoves               10485760    6608517   63%         20%
"""
    assert parse_fs_listquota(output) == (10485760 - 6608517) * 1024


def test_a_volume_without_a_limit_is_not_mistaken_for_a_full_one():
    output = """Volume Name                    Quota       Used %Used   Partition
work.dgenoves                no limit   6608517   n/a         20%
"""
    assert parse_fs_listquota(output) is None
    assert parse_fs_listquota("") is None


def test_a_dataset_that_does_not_fit_is_refused_before_anything_is_written(tmp_path):
    # 2400 files of a 159-feature vector, the size of the default experiment, against
    # the 3.9 GB an AFS home actually had left when this was written.
    with pytest.raises(RuntimeError) as err:
        check_output_space(str(tmp_path), n_files=2400, vlen=159, free=int(3.9e9))

    message = str(err.value)
    assert "GB" in message and "eos_data_dir" in message, "say how much, and where instead"


def test_a_dataset_that_fits_goes_ahead(tmp_path):
    check_output_space(str(tmp_path), n_files=6, vlen=159, free=int(3.9e9))
    check_output_space(str(tmp_path), n_files=0, vlen=159, free=0)  # nothing left to write


# ---------------------------------------------------------------------------
# Counting a dataset without opening every shard
# ---------------------------------------------------------------------------


def write_shards(directory, rows_per_shard):
    os.makedirs(directory, exist_ok=True)
    for i, rows in enumerate(rows_per_shard):
        np.save(os.path.join(directory, f"shard{i}_x.npy"), np.zeros((rows, 3), dtype=np.float32))
        np.save(os.path.join(directory, f"shard{i}_y.npy"), np.zeros(rows, dtype=np.int64))


def test_counts_are_correct_and_cached(tmp_path):
    write_shards(tmp_path, [5, 7, 11])

    assert shard_rows(str(tmp_path)) == [("shard0_x.npy", 5), ("shard1_x.npy", 7),
                                         ("shard2_x.npy", 11)]
    assert count_split_events(str(tmp_path)) == (23, 3)
    assert (tmp_path / SHARD_ROWS_CACHE).exists()


def test_a_cached_count_is_used_without_opening_the_file(tmp_path, monkeypatch):
    """The point of the cache: the second check opens nothing."""
    write_shards(tmp_path, [5, 7])
    shard_rows(str(tmp_path))

    def refuse(*args, **kwargs):
        raise AssertionError("a header was read although the cache was valid")

    monkeypatch.setattr(np, "load", refuse)
    assert count_split_events(str(tmp_path)) == (12, 2)


def test_a_changed_shard_is_counted_again(tmp_path):
    """A stale count is worse than a slow one, so size and mtime must both match."""
    write_shards(tmp_path, [5, 7])
    shard_rows(str(tmp_path))

    time.sleep(0.01)
    np.save(tmp_path / "shard1_x.npy", np.zeros((3, 3), dtype=np.float32))  # rewritten short

    assert count_split_events(str(tmp_path)) == (8, 2)
    cached = json.loads((tmp_path / SHARD_ROWS_CACHE).read_text())
    assert cached["shard1_x.npy"][2] == 3


def test_a_new_shard_is_noticed_and_a_removed_one_forgotten(tmp_path):
    write_shards(tmp_path, [5, 7])
    shard_rows(str(tmp_path))

    os.remove(tmp_path / "shard0_x.npy")
    np.save(tmp_path / "shard9_x.npy", np.zeros((4, 3), dtype=np.float32))

    assert shard_rows(str(tmp_path)) == [("shard1_x.npy", 7), ("shard9_x.npy", 4)]


def test_a_read_only_dataset_is_counted_without_a_cache(tmp_path):
    """Someone else's dataset: count it, and do not fail because the cache cannot be written."""
    write_shards(tmp_path, [5, 7])
    os.chmod(tmp_path, 0o555)
    try:
        assert count_split_events(str(tmp_path)) == (12, 2)
        assert not (tmp_path / SHARD_ROWS_CACHE).exists()
    finally:
        os.chmod(tmp_path, 0o755)


def test_a_corrupt_cache_is_ignored(tmp_path):
    write_shards(tmp_path, [5])
    (tmp_path / SHARD_ROWS_CACHE).write_text("{not json")

    assert count_split_events(str(tmp_path)) == (5, 1)
