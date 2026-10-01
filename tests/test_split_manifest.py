"""Tests for the split manifest: the choice of which Parquet file goes where.

The manifest decides what a dataset contains, so the properties checked here are
the ones that make two runs comparable: the same seed gives the same files, a
different seed gives different ones, and one class is unaffected by the presence
of the others.
"""

import json
import os
import tempfile

from src.data.utils import (
    check_shards_match_manifest,
    class_seed,
    load_split_manifest,
    make_split_manifest,
    resolve_split_manifest,
    save_split_manifest,
    split_manifest_meta,
)

SPLIT_COUNTS = [50, 20, 20]


def fake_filelist(n_classes: int = 3, n_files: int = 12, n_events: int = 10) -> dict:
    """{folder: {filename: events}}, the shape of file_event_counts.json."""
    return {
        f"class_{c}": {f"class_{c}-RS{i:04d}.parquet": n_events for i in range(n_files)}
        for c in range(n_classes)
    }


def files_of(manifest: dict, folder: str) -> dict:
    return {split: list(files) for split, files in manifest[folder].items()}


def test_same_seed_same_manifest():
    filelist = fake_filelist()
    folders = list(filelist)

    first = make_split_manifest(filelist, SPLIT_COUNTS, folders, seed=1)
    second = make_split_manifest(filelist, SPLIT_COUNTS, folders, seed=1)

    assert first == second


def test_different_seed_different_manifest():
    filelist = fake_filelist(n_files=40)
    folders = list(filelist)

    first = make_split_manifest(filelist, SPLIT_COUNTS, folders, seed=1)
    second = make_split_manifest(filelist, SPLIT_COUNTS, folders, seed=2)

    assert first != second


def test_class_choice_independent_of_the_other_classes():
    """Dropping a class must not move the files of the classes that remain.

    With one generator shared by every class, removing `class_0` shifted the draw
    of `class_1` and `class_2`, so a dataset could not be extended or reduced
    without rebuilding all of it.
    """
    filelist = fake_filelist()

    all_classes = make_split_manifest(filelist, SPLIT_COUNTS, list(filelist), seed=1)
    fewer = make_split_manifest(filelist, SPLIT_COUNTS, ["class_1", "class_2"], seed=1)

    for folder in ("class_1", "class_2"):
        assert files_of(all_classes, folder) == files_of(fewer, folder)


def test_order_of_the_event_count_json_does_not_matter():
    """The same files in a different JSON order must give the same manifest."""
    filelist = fake_filelist(n_classes=1)
    reversed_filelist = {
        folder: dict(reversed(list(files.items()))) for folder, files in filelist.items()
    }

    assert make_split_manifest(filelist, SPLIT_COUNTS, ["class_0"], seed=1) == make_split_manifest(
        reversed_filelist, SPLIT_COUNTS, ["class_0"], seed=1
    )


def test_targets_are_met_without_waste():
    """Each split gets just enough files to reach its target, and no more."""
    filelist = fake_filelist(n_classes=1, n_files=40, n_events=10)

    manifest = make_split_manifest(filelist, SPLIT_COUNTS, ["class_0"], seed=1)
    buckets = manifest["class_0"]

    for split, target in zip(("train", "val", "test"), SPLIT_COUNTS):
        events = 10 * len(buckets[split])
        assert events >= target
        assert events - 10 < target, f"{split} took a file more than it needed"

    chosen = [f for split in buckets.values() for f in split]
    assert len(chosen) == len(set(chosen)), "a file was used in more than one split"


def test_legacy_strategy_couples_the_classes():
    """The behaviour `legacy` exists to reproduce, stated as a test.

    Under the shared generator, dropping a class moves the files of the classes
    that remain. That is why `per_class` is the default; `legacy` is only for
    rebuilding a dataset produced before it existed.
    """
    filelist = fake_filelist()

    all_classes = make_split_manifest(filelist, SPLIT_COUNTS, list(filelist), seed=1, strategy="legacy")
    fewer = make_split_manifest(filelist, SPLIT_COUNTS, ["class_1", "class_2"], seed=1, strategy="legacy")

    assert files_of(all_classes, "class_1") != files_of(fewer, "class_1")


def test_legacy_strategy_is_reproducible():
    filelist = fake_filelist()
    folders = list(filelist)

    first = make_split_manifest(filelist, SPLIT_COUNTS, folders, seed=1, strategy="legacy")
    second = make_split_manifest(filelist, SPLIT_COUNTS, folders, seed=1, strategy="legacy")

    assert first == second
    assert first != make_split_manifest(filelist, SPLIT_COUNTS, folders, seed=1)


def test_legacy_strategy_matches_the_original_code():
    """Pinned output of the pre-existing implementation, so `legacy` cannot drift.

    Produced by running `make_split_manifest` from commit 13448e6 on the filelist
    `fake_filelist()` builds. If this test fails, `legacy` no longer rebuilds the
    datasets it exists to rebuild.
    """
    expected = {
        "class_0": {
            "train": ["class_0-RS0008.parquet", "class_0-RS0011.parquet", "class_0-RS0004.parquet",
                      "class_0-RS0007.parquet", "class_0-RS0005.parquet"],
            "val": ["class_0-RS0000.parquet", "class_0-RS0001.parquet"],
            "test": ["class_0-RS0009.parquet", "class_0-RS0002.parquet"],
        },
        "class_1": {
            "train": ["class_1-RS0001.parquet", "class_1-RS0003.parquet", "class_1-RS0006.parquet",
                      "class_1-RS0000.parquet", "class_1-RS0007.parquet"],
            "val": ["class_1-RS0004.parquet", "class_1-RS0005.parquet"],
            "test": ["class_1-RS0008.parquet", "class_1-RS0009.parquet"],
        },
        "class_2": {
            "train": ["class_2-RS0003.parquet", "class_2-RS0011.parquet", "class_2-RS0002.parquet",
                      "class_2-RS0000.parquet", "class_2-RS0009.parquet"],
            "val": ["class_2-RS0006.parquet", "class_2-RS0007.parquet"],
            "test": ["class_2-RS0010.parquet", "class_2-RS0001.parquet"],
        },
    }

    produced = make_split_manifest(
        fake_filelist(), SPLIT_COUNTS, ["class_0", "class_1", "class_2"], seed=1, strategy="legacy"
    )

    assert produced == expected


def test_unknown_strategy_is_refused():
    filelist = fake_filelist(n_classes=1)
    try:
        make_split_manifest(filelist, SPLIT_COUNTS, ["class_0"], strategy="per-class")
    except ValueError as err:
        assert "per-class" in str(err)
    else:
        raise AssertionError("a misspelled strategy must not fall back to a default")


def test_class_seed_is_stable_across_processes():
    """Hard-coded values: built-in hash() is salted per process, hashlib is not."""
    assert class_seed("class_0", 42) == 17_329_860_780_148_671_489
    assert class_seed("class_0", 43) != class_seed("class_0", 42)
    assert class_seed("class_1", 42) != class_seed("class_0", 42)


# ---------------------------------------------------------------------------
# Storing the manifest, and refusing to mix two selections
# ---------------------------------------------------------------------------


def test_manifest_round_trip_keeps_metadata_out_of_the_folders():
    """`_meta` is stored inside the file but must never look like a class."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "split_manifest.json")
        built = make_split_manifest(fake_filelist(n_classes=2), SPLIT_COUNTS, ["class_0", "class_1"])
        meta = split_manifest_meta(seed=1, strategy="per_class", split_counts=SPLIT_COUNTS)

        save_split_manifest(path, built, meta)
        folders, loaded_meta = load_split_manifest(path)

        assert folders == built
        assert loaded_meta == meta
        assert "_meta" in json.load(open(path))


def test_existing_manifest_is_reused_and_not_rewritten():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "split_manifest.json")
        filelist = fake_filelist(n_classes=2)
        folders = ["class_0", "class_1"]

        first = resolve_split_manifest(path, folders, SPLIT_COUNTS, seed=1, global_filelist=filelist)
        written_at = os.path.getmtime(path)

        second = resolve_split_manifest(path, folders, SPLIT_COUNTS, seed=1, global_filelist=filelist)

        assert first == second
        assert os.path.getmtime(path) == written_at, "an existing manifest must not be rewritten"


def test_a_new_class_extends_the_manifest_without_moving_the_others():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "split_manifest.json")
        filelist = fake_filelist(n_classes=3)

        two = resolve_split_manifest(path, ["class_0", "class_1"], SPLIT_COUNTS, seed=1,
                                     global_filelist=filelist)
        three = resolve_split_manifest(path, ["class_0", "class_1", "class_2"], SPLIT_COUNTS, seed=1,
                                       global_filelist=filelist)

        assert set(three) == {"class_0", "class_1", "class_2"}
        for folder in ("class_0", "class_1"):
            assert three[folder] == two[folder]


def test_changing_the_seed_on_an_existing_dataset_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "split_manifest.json")
        filelist = fake_filelist(n_classes=1)

        resolve_split_manifest(path, ["class_0"], SPLIT_COUNTS, seed=1, global_filelist=filelist)

        for changed in ({"seed": 2}, {"strategy": "legacy"}, {"split_counts": [10, 10, 10]}):
            kwargs = {"seed": 1, "split_counts": SPLIT_COUNTS, **changed}
            try:
                resolve_split_manifest(path, ["class_0"], global_filelist=filelist, **kwargs)
            except ValueError as err:
                assert "data.label" in str(err)
            else:
                raise AssertionError(f"{changed} should not be accepted on an existing manifest")


def test_a_manifest_without_metadata_is_still_usable():
    """Datasets built before the metadata existed must keep working."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "split_manifest.json")
        filelist = fake_filelist(n_classes=1)
        old_style = make_split_manifest(filelist, SPLIT_COUNTS, ["class_0"])
        with open(path, "w") as f:
            json.dump(old_style, f)

        assert resolve_split_manifest(path, ["class_0"], SPLIT_COUNTS, seed=99,
                                      global_filelist=filelist) == old_style


def test_shards_from_another_selection_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        filelist = fake_filelist(n_classes=1)
        manifest = make_split_manifest(filelist, SPLIT_COUNTS, ["class_0"])

        train_dir = os.path.join(tmp, "train", "class_0")
        os.makedirs(train_dir)
        for fname in manifest["class_0"]["train"]:
            open(os.path.join(train_dir, fname.replace(".parquet", "_x.npy")), "w").close()

        check_shards_match_manifest(tmp, manifest)  # consistent: no error

        open(os.path.join(train_dir, "class_0-RS9999_x.npy"), "w").close()
        try:
            check_shards_match_manifest(tmp, manifest)
        except ValueError as err:
            assert "RS9999" in str(err)
        else:
            raise AssertionError("a shard outside the manifest must stop the run")


def test_shard_check_only_warns_for_a_manifest_without_metadata():
    """A dataset built before the rule existed keeps working, with a warning.

    Measured on the datasets on EOS: one of them holds QCD shards its manifest no
    longer lists, so making this an error would block runs on data already taken.
    """
    with tempfile.TemporaryDirectory() as tmp:
        filelist = fake_filelist(n_classes=1)
        manifest = make_split_manifest(filelist, SPLIT_COUNTS, ["class_0"])

        train_dir = os.path.join(tmp, "train", "class_0")
        os.makedirs(train_dir)
        open(os.path.join(train_dir, "class_0-RS9999_x.npy"), "w").close()

        # No metadata: a warning, and the run goes on.
        with open(os.path.join(tmp, "split_manifest.json"), "w") as f:
            json.dump(manifest, f)
        check_shards_match_manifest(tmp, manifest)

        # Metadata present, so the dataset was built under the rule: an error.
        save_split_manifest(
            os.path.join(tmp, "split_manifest.json"),
            manifest,
            split_manifest_meta(seed=1, strategy="per_class", split_counts=SPLIT_COUNTS),
        )
        try:
            check_shards_match_manifest(tmp, manifest)
        except ValueError as err:
            assert "RS9999" in str(err)
        else:
            raise AssertionError("with metadata, a foreign shard must stop the run")
