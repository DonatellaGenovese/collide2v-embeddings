"""Tests for the two checks that run before vectorisation starts.

Both exist to turn a failure that used to arrive late — inside a batch job, after
reading files for an hour, or not at all — into one that arrives immediately and says
what is wrong.
"""

import json
import os
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.utils import (
    build_feature_map_dict,
    check_columns_present,
    check_feature_map_matches,
    save_feature_map,
)

CONFIG = {
    "jets": {
        "cols": ["FullReco_JetPuppiAK4_PT", "FullReco_JetPuppiAK4_Eta",
                 "FullReco_JetPuppiAK4_NNeutrals"],
        "topk": 4,
        "count": True,
    },
    "puppi_met": {"cols": ["FullReco_PUPPIMET_MET"], "topk": None, "count": False},
}
ALL_COLS = [c for group in CONFIG.values() for c in group["cols"]]
MANIFEST = {"class_a": {"train": ["a1.parquet"], "val": ["a2.parquet"], "test": []}}


def write_parquet(path, columns):
    table = pa.table({c: pa.array([[1.0], [2.0]], type=pa.large_list(pa.float32()))
                      for c in columns})
    pq.write_table(table, path)


def test_all_columns_present_is_silent():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "class_a"))
        for name in ("a1.parquet", "a2.parquet"):
            write_parquet(os.path.join(tmp, "class_a", name), ALL_COLS)

        check_columns_present(tmp, MANIFEST, ALL_COLS)


def test_a_missing_column_is_named_before_any_work():
    """What reading the Hugging Face copy with the EOS feature set looks like."""
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "class_a"))
        for name in ("a1.parquet", "a2.parquet"):
            write_parquet(os.path.join(tmp, "class_a", name),
                          [c for c in ALL_COLS if "NNeutrals" not in c])

        with pytest.raises(ValueError) as err:
            check_columns_present(tmp, MANIFEST, ALL_COLS)

        assert "FullReco_JetPuppiAK4_NNeutrals" in str(err.value)
        assert "collide2v_common" in str(err.value), "the message should say what to use instead"


def test_only_the_first_file_of_each_class_is_opened():
    """The check must stay cheap: the second file is deliberately unreadable."""
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "class_a"))
        write_parquet(os.path.join(tmp, "class_a", "a1.parquet"), ALL_COLS)
        with open(os.path.join(tmp, "class_a", "a2.parquet"), "w") as f:
            f.write("not a parquet file")

        check_columns_present(tmp, MANIFEST, ALL_COLS)


def test_a_file_in_the_manifest_but_not_on_disk_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "class_a"))

        with pytest.raises(FileNotFoundError):
            check_columns_present(tmp, MANIFEST, ALL_COLS)


def test_an_unchanged_feature_map_is_accepted():
    with tempfile.TemporaryDirectory() as tmp:
        save_feature_map(CONFIG, tmp, vlen=0)

        check_feature_map_matches(tmp, CONFIG)


def test_no_feature_map_yet_is_accepted():
    with tempfile.TemporaryDirectory() as tmp:
        check_feature_map_matches(tmp, CONFIG)


def test_changing_the_feature_set_without_changing_the_label_is_refused():
    """The case switching the default feature set creates.

    The manifest records which files a dataset holds, not which columns were taken from
    them, so without this check a dataset built with the EOS feature set would quietly
    gain shards built with the common one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        save_feature_map(CONFIG, tmp, vlen=0)
        fewer = {
            "jets": {"cols": CONFIG["jets"]["cols"][:2], "topk": 4, "count": True},
            "puppi_met": CONFIG["puppi_met"],
        }

        with pytest.raises(ValueError) as err:
            check_feature_map_matches(tmp, fewer)

        message = str(err.value)
        assert "jets" in message
        assert "data.label" in message, "the message should say what to do"


def test_the_stored_feature_map_is_what_the_config_describes():
    with tempfile.TemporaryDirectory() as tmp:
        save_feature_map(CONFIG, tmp, vlen=0)
        with open(os.path.join(tmp, "feature_map.json")) as f:
            stored = json.load(f)

        assert stored == build_feature_map_dict(CONFIG)
        # 3 columns x topk 4 + 1 count, then the scalar
        assert stored["jets"]["end"] == 13
        assert stored["puppi_met"]["start"] == 13
        assert stored["puppi_met"]["end"] == 14
