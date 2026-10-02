"""End-to-end tests of vectorisation, preprocessing and reading, on fake data.

Everything these tests need is written by the tests themselves: a handful of tiny
Parquet files with the column names the pipeline recognises. Nothing touches EOS,
so they run on a laptop and in CI, in seconds, and they are the quickest way to
check that an environment is set up correctly.

What they assert is the property the dataset rests on: given the same inputs and
the same seed, the pipeline writes the same bytes, whatever the machine does with
workers and whatever order the files come back in.
"""

import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.datasets import LocalVectorDataset
from src.data.utils import (
    MANIFEST_NAME,
    compute_vlen,
    get_all_cols,
    has_enough_events,
    resolve_split_manifest,
    vectorize_to_local,
)
from src.preprocessing.preprocess import PreprocessingPipeline

# Two classes, three files each, so that a split boundary falls inside a class.
CLASSES = ["signal", "background"]
FOLDERS = {"signal": "fake_signal", "background": "fake_background"}
FILES_PER_CLASS = 3
EVENTS_PER_FILE = 40

# Column names matter: the preprocessing infers the transform group from the
# suffix (_PT, _Eta, _Phi, _Mass, _BTag, _MET), so fake data has to use them too.
DATASETS_CONFIG = {
    "jets": {
        "cols": [
            "FullReco_JetPuppiAK4_PT",
            "FullReco_JetPuppiAK4_Eta",
            "FullReco_JetPuppiAK4_Phi",
            "FullReco_JetPuppiAK4_Mass",
        ],
        "topk": 4,
        "count": True,
    },
    "puppi_met": {
        "cols": ["FullReco_PUPPIMET_MET", "FullReco_PUPPIMET_Phi"],
        "topk": None,
        "count": False,
    },
}
PREPROCESS_CFG = {
    "enabled": True,
    "mode": "fit_and_apply",
    "fit_num_files_per_class": 1,
    "feature_transforms": {"pt": "log1p", "eta": "identity", "phi": "trig",
                           "mass": "log1p", "met": "log1p", "count": "identity",
                           "other": "identity"},
    "feature_normalizations": {"pt": "robust", "eta": "robust", "phi": "none",
                               "mass": "robust", "met": "robust", "count": "minmax",
                               "other": "robust"},
}
SPLITS = [40, 40, 40]   # one file per split, with three files per class


def write_fake_dataset(dataset_dir):
    """Parquet files shaped like COLLIDE-2V: one list per object collection.

    Even the scalars are lists of one element there, as `FullReco_PUPPIMET_MET` is
    on EOS. The two classes are deliberately written with different Arrow types:
    `signal` as EOS does it, `large_list<halffloat>`, and `background` as another
    tool would, `list<float>`, whose offsets are 32-bit. Both must be readable.
    """
    rng = np.random.default_rng(0)
    filelist = {}

    for class_index, cname in enumerate(CLASSES):
        folder = FOLDERS[cname]
        os.makedirs(os.path.join(dataset_dir, folder))
        filelist[folder] = {}

        if class_index == 0:
            list_type, value_type = pa.large_list, pa.float16()
        else:
            list_type, value_type = pa.list_, pa.float32()

        for file_index in range(FILES_PER_CLASS):
            n_objects = rng.integers(0, 7, size=EVENTS_PER_FILE)  # some events have none
            columns = {}
            for col in DATASETS_CONFIG["jets"]["cols"]:
                columns[col] = pa.array(
                    [rng.normal(50 + 10 * class_index, 20, size=int(k)).astype(np.float16)
                     for k in n_objects],
                    type=list_type(value_type),
                )
            for col, draw in (
                ("FullReco_PUPPIMET_MET", lambda: rng.uniform(0, 200)),
                ("FullReco_PUPPIMET_Phi", lambda: rng.uniform(-np.pi, np.pi)),
            ):
                columns[col] = pa.array(
                    [np.array([draw()], dtype=np.float16) for _ in range(EVENTS_PER_FILE)],
                    type=list_type(value_type),
                )

            name = f"{folder}-NEVENT{EVENTS_PER_FILE}-RS{file_index:04d}.parquet"
            pq.write_table(pa.table(columns), os.path.join(dataset_dir, folder, name))
            filelist[folder][name] = EVENTS_PER_FILE

    return filelist


def run_pipeline(root, dataset_dir, filelist, seed=3, drop_empty_events=False):
    """Vectorise, then preprocess, into `root`. Returns the paths it used."""
    paths = {
        "eos_vec_dir": os.path.join(root, "vectorized"),
        "tmp_vec_dir": os.path.join(root, "tmp_vec"),
        "eos_preproc_dir": os.path.join(root, "preprocessed"),
        "tmp_preproc_dir": os.path.join(root, "tmp_preproc"),
    }
    os.makedirs(paths["eos_vec_dir"], exist_ok=True)

    manifest = resolve_split_manifest(
        manifest_path=os.path.join(paths["eos_vec_dir"], MANIFEST_NAME),
        include_folders=[FOLDERS[c] for c in CLASSES],
        split_counts=SPLITS,
        seed=seed,
        global_filelist=filelist,
    )
    vectorize_to_local(
        base_dir=dataset_dir,
        config=DATASETS_CONFIG,
        class_names=CLASSES,
        folder_map=FOLDERS,
        labels_map={c: i for i, c in enumerate(CLASSES)},
        all_cols=get_all_cols(DATASETS_CONFIG),
        vlen=compute_vlen(DATASETS_CONFIG),
        tmp_vec_dir=paths["tmp_vec_dir"],
        eos_vec_dir=paths["eos_vec_dir"],
        split_counts=SPLITS,
        split_manifest=manifest,
        seed=seed,
        drop_empty_events=drop_empty_events,
    )
    PreprocessingPipeline(
        paths=paths,
        preprocess_cfg=PREPROCESS_CFG,
        process_to_folder=FOLDERS,
        class_order=CLASSES,
        device="cpu",
    ).run()
    return paths


def shard_contents(root, stage="preprocessed"):
    """{relative path: array} for every shard, for comparing two runs byte for byte."""
    contents = {}
    base = os.path.join(root, stage)
    for split in ("train", "val", "test"):
        for folder in sorted(FOLDERS.values()):
            d = os.path.join(base, split, folder)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                if name.endswith(".npy"):
                    contents[f"{split}/{folder}/{name}"] = np.load(os.path.join(d, name))
    return contents


@pytest.fixture(scope="module")
def fake_dataset(tmp_path_factory):
    dataset_dir = tmp_path_factory.mktemp("collide_fake")
    filelist = write_fake_dataset(str(dataset_dir))
    return str(dataset_dir), filelist


@pytest.fixture(scope="module")
def two_runs(fake_dataset, tmp_path_factory):
    """The same pipeline, same seed, run twice into two separate directories."""
    dataset_dir, filelist = fake_dataset
    first = str(tmp_path_factory.mktemp("run_a"))
    second = str(tmp_path_factory.mktemp("run_b"))
    run_pipeline(first, dataset_dir, filelist)
    run_pipeline(second, dataset_dir, filelist)
    return first, second


def test_two_runs_write_identical_shards(two_runs):
    first, second = two_runs

    for stage in ("vectorized", "preprocessed"):
        a, b = shard_contents(first, stage), shard_contents(second, stage)
        assert sorted(a) == sorted(b), f"{stage}: different shards on disk"
        assert a, f"{stage}: no shards were written"
        for name in a:
            assert np.array_equal(a[name], b[name]), f"{stage}/{name} differs between runs"


def test_two_runs_agree_on_the_manifest_and_the_statistics(two_runs):
    first, second = two_runs

    manifest_a = json.load(open(os.path.join(first, "vectorized", MANIFEST_NAME)))
    manifest_b = json.load(open(os.path.join(second, "vectorized", MANIFEST_NAME)))
    assert manifest_a == manifest_b
    assert manifest_a["_meta"] == {"seed": 3, "manifest_strategy": "per_class",
                                  "split_counts": SPLITS}

    stats_a = json.load(open(os.path.join(first, "preprocessed", "norm_stats.json")))
    stats_b = json.load(open(os.path.join(second, "preprocessed", "norm_stats.json")))
    assert stats_a == stats_b


def test_a_second_vectorisation_changes_nothing(two_runs, fake_dataset):
    """Rerunning must be a no-op, not a second pass that rewrites the shards."""
    first, _ = two_runs
    dataset_dir, filelist = fake_dataset

    before = {name: os.path.getmtime(path)
              for name, path in _shard_paths(first, "vectorized").items()}
    run_pipeline(first, dataset_dir, filelist)
    after = {name: os.path.getmtime(path)
             for name, path in _shard_paths(first, "vectorized").items()}

    assert before == after


def _shard_paths(root, stage):
    paths = {}
    base = os.path.join(root, stage)
    for split in ("train", "val", "test"):
        for folder in sorted(FOLDERS.values()):
            d = os.path.join(base, split, folder)
            if os.path.isdir(d):
                for name in sorted(os.listdir(d)):
                    paths[f"{split}/{folder}/{name}"] = os.path.join(d, name)
    return paths


def test_a_missing_preprocessed_shard_is_filled_in_without_refitting(two_runs, fake_dataset):
    """What a batch job dying half way through leaves behind, and the recovery.

    The statistics must not be refitted: they are part of what the dataset is, and
    fitting them again on a different set of files would rescale the new shard
    against the old ones.
    """
    first, second = two_runs
    dataset_dir, filelist = fake_dataset

    target = _shard_paths(first, "preprocessed")["test/fake_signal/" + sorted(
        os.listdir(os.path.join(first, "preprocessed", "test", "fake_signal"))
    )[0]]
    expected = np.load(target)
    stats_path = os.path.join(first, "preprocessed", "norm_stats.json")
    stats_mtime = os.path.getmtime(stats_path)
    os.remove(target)

    run_pipeline(first, dataset_dir, filelist)

    assert os.path.exists(target), "the missing shard was not written again"
    assert np.array_equal(np.load(target), expected), "the refilled shard differs"
    assert os.path.getmtime(stats_path) == stats_mtime, "the statistics were refitted"


@pytest.mark.parametrize("num_workers", [0, 2, 3])
def test_the_loader_reads_the_same_events_whatever_the_workers(two_runs, num_workers):
    first, _ = two_runs
    per_class = 25

    def sample(workers):
        ds = LocalVectorDataset(
            os.path.join(first, "preprocessed", "test"),
            per_class_limit=per_class,
            shuffle_file_order=True,
            classnames=CLASSES,
            folder_map=FOLDERS,
            seed=3,
        )
        rows, labels = [], []
        for x, y in DataLoader(ds, batch_size=8, num_workers=workers):
            rows.extend(x.sum(dim=1).tolist())
            labels.extend(y.tolist())
        return sorted(rows), labels

    reference_rows, reference_labels = sample(0)
    rows, labels = sample(num_workers)

    assert len(reference_labels) == 2 * per_class
    assert rows == reference_rows
    assert sorted(labels) == sorted(reference_labels)
    assert labels.count(0) == per_class and labels.count(1) == per_class


def test_the_dataset_passes_the_event_check(two_runs):
    first, _ = two_runs

    assert has_enough_events(os.path.join(first, "preprocessed"), SPLITS, CLASSES, FOLDERS)


def test_preprocessing_widens_the_vector_as_the_feature_map_says(two_runs):
    """phi becomes (sin, cos), so the preprocessed shards are wider than the raw ones."""
    first, _ = two_runs

    raw = shard_contents(first, "vectorized")
    done = shard_contents(first, "preprocessed")
    fm = json.load(open(os.path.join(first, "preprocessed", "feature_map.json")))
    width = max(section["end"] for section in fm.values())

    x_name = next(name for name in sorted(done) if name.endswith("_x.npy"))
    assert done[x_name].shape[1] == width
    assert done[x_name].shape[1] > raw[x_name].shape[1]
    assert done[x_name].shape[0] == raw[x_name].shape[0]
    assert np.isfinite(done[x_name]).all(), "preprocessing produced NaN or inf"


def test_dropping_empty_events_removes_exactly_those(fake_dataset, tmp_path_factory):
    """The filter is a physics choice, so what it removes is pinned by a test.

    The fake files are written with some events holding no objects at all, which is
    what the filter is for: with it off they are kept as all-zero rows, with it on they
    are gone and nothing else is.
    """
    dataset_dir, filelist = fake_dataset
    kept_root = str(tmp_path_factory.mktemp("with_empty"))
    dropped_root = str(tmp_path_factory.mktemp("without_empty"))

    run_pipeline(kept_root, dataset_dir, filelist, drop_empty_events=False)
    run_pipeline(dropped_root, dataset_dir, filelist, drop_empty_events=True)

    kept = shard_contents(kept_root, "vectorized")
    dropped = shard_contents(dropped_root, "vectorized")
    fm = json.load(open(os.path.join(kept_root, "vectorized", "feature_map.json")))
    jet_pt_columns = [fm["jets"]["start"] + i * len(fm["jets"]["columns"])
                      for i in range(fm["jets"]["topk"])]

    total_empty = 0
    for name, X in kept.items():
        if not name.endswith("_x.npy"):
            continue
        empty = (X[:, jet_pt_columns] == 0).all(axis=1)
        total_empty += int(empty.sum())
        assert len(dropped[name]) == len(X) - int(empty.sum())
        assert not (dropped[name][:, jet_pt_columns] == 0).all(axis=1).any()

    assert total_empty > 0, "the fake dataset should contain events with no objects"
