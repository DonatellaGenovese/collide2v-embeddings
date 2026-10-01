import hashlib
import json
import os
import random
import shutil

from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf


# ============================================================
# CONFIG DERIVED UTILITIES
# ============================================================


def get_all_cols(config: dict):
    """Return a flat list of all columns used across all dataset groups."""
    cols = []
    for spec in config.values():
        cols.extend(spec["cols"])
    return cols


def compute_vlen(config: dict) -> int:
    """Compute the total flattened vector length given a dataset config."""
    total = 0
    for spec in config.values():
        ncols = len(spec["cols"])
        topk = spec["topk"]
        if topk is None:
            total += ncols  # scalars
        else:
            total += ncols * topk
        if spec.get("count", False):
            total += 1
    return total


# ============================================================
# FEATURE PACKING HELPERS
# ============================================================


def fill_masked(array: np.ndarray, fill: float) -> np.ndarray:
    """A plain array, with masked entries replaced by the padding value.

    `ak.to_numpy` returns a masked array when the Arrow field is nullable. The
    columns on EOS are declared `not null`, so this never came up there, but a
    Parquet file written anywhere else is nullable by default, and np.save then
    fails with `MaskedArray.tofile() not implemented yet` — after the whole file
    has been read.
    """
    return np.ma.filled(array, fill) if np.ma.isMaskedArray(array) else array


def _pack_topk_batch(pt, *others, k: int, fill: float):
    """Picks top-k objects by first feature (e.g. pt).

    Returns: np.ndarray of shape (n_events, k, n_features)
    """
    arrays = (pt,) + others
    expanded = [arr[:, :, None] for arr in arrays]
    stacked = ak.concatenate(expanded, axis=2)

    # convert and sort
    stacked = ak.values_astype(stacked, np.float32)
    stacked = stacked[ak.argsort(stacked[:, :, 0], axis=1, ascending=False)]

    # take top-k
    topk = stacked[:, :k, :]
    n_features = len(arrays)

    # pad / fill
    topk = ak.pad_none(topk, k, axis=1)
    fill_list = [fill] * n_features
    topk = ak.fill_none(topk, fill_list, axis=1)

    return fill_masked(ak.to_numpy(topk), fill)


def _pack_leading_batch(pt, *others, fill: float):
    """Picks top-1 object (highest pt).

    Returns: np.ndarray of shape (n_events, 1, n_features)
    """
    arrays = (pt,) + others
    expanded = [arr[:, :, None] for arr in arrays]
    stacked = ak.concatenate(expanded, axis=2)

    stacked = ak.values_astype(stacked, np.float32)
    leading = stacked[ak.argmax(stacked[:, :, 0], axis=1, keepdims=True)]

    n_features = len(arrays)
    leading = ak.pad_none(leading, 1, axis=1)
    fill_list = [fill] * n_features
    leading = ak.fill_none(leading, fill_list, axis=1)

    return fill_masked(ak.to_numpy(leading), fill)


# ============================================================
# BATCH VECTOR CONSTRUCTION
# ============================================================


def to_awkward_column(column):
    """One Arrow column as an awkward array, with 32-bit list offsets widened.

    The dataset on EOS stores these columns as `large_list`, whose offsets are
    64-bit, and that is the only type this pipeline ever saw. A Parquet file
    written by another tool — or by a test — may use `list` instead, and awkward
    then fails inside argsort with `awkward_NumpyArray_rearrange_shifted` and a
    tuple of dtypes for a message. Widening the offsets first costs nothing and
    keeps both kinds of file readable.
    """
    if pa.types.is_list(column.type):
        column = column.cast(pa.large_list(column.type.value_type))
    return ak.from_arrow(column)


def build_vectors_batch(batch: dict, config: dict, fill: float = 0.0) -> np.ndarray:
    """
    batch: dict mapping column name → awkward.Array
    config: dataset config (e.g. cfg.data.datasets_config)
    fill: fill value for missing entries

    Returns: np.ndarray of shape (n_events, VLEN)
    """
    features = []

    for name, spec in config.items():
        cols = spec["cols"]
        topk = spec["topk"]

        if topk is None:
            # scalars (e.g. MET)
            vals = [
                fill_masked(ak.to_numpy(ak.fill_none(batch[c], fill)), fill).reshape(-1, 1)
                for c in cols
            ]
            group = np.concatenate(vals, axis=1)

        elif topk == 1:
            arrays = [batch[c] for c in cols]
            group = _pack_leading_batch(*arrays, fill=fill).reshape(len(arrays[0]), -1)

        else:
            arrays = [batch[c] for c in cols]
            group = _pack_topk_batch(*arrays, k=topk, fill=fill).reshape(len(arrays[0]), -1)

        features.append(group)

        # optional count feature
        if spec.get("count", False):
            nobj = ak.num(batch[cols[0]], axis=1).to_numpy().reshape(-1, 1)
            features.append(nobj)

    # concatenate everything into flat vector
    return np.concatenate(features, axis=1)


# ============================================================
# FEATURE MAP SAVING
# ============================================================


def build_feature_map_dict(config) -> dict:
    """Where each group lands in the flat vector, from a datasets_config.

    Takes a Hydra config or a plain dict: a plain dict is what a test, a notebook
    or a script that does not go through Hydra has, and the function used to
    accept only the former.
    """
    feature_map = {}
    offset = 0

    config_dict = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)

    for group_name, cfg in config_dict.items():
        cols = cfg["cols"]
        topk = cfg["topk"]
        count = cfg.get("count", False)

        if topk is None:
            size = len(cols)
        else:
            size = len(cols) * topk
        if count:
            size += 1

        feature_map[group_name] = {
            "start": int(offset),
            "end": int(offset + size),
            "columns": cols,
            "topk": topk,
            "count": count,
        }
        offset += size

    return feature_map


def save_feature_map(config, out_dir: str, vlen: int):
    """Save a feature_map.json describing flattened layout."""
    feature_map = build_feature_map_dict(config)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "feature_map.json"), "w") as f:
        json.dump(feature_map, f, indent=2)

    print(f"✓ feature_map.json saved → {out_dir}")


# ============================================================
# FILE + STORAGE UTILITIES
# ============================================================


def move_to_eos(local_dir: str, eos_dir: str):
    """Move a directory or file from a local tmpdir to EOS target."""
    os.makedirs(os.path.dirname(eos_dir), exist_ok=True)
    try:
        shutil.move(local_dir, eos_dir)
    except Exception as e:
        print(f"⚠️ Move failed: {e}")

def load_global_filelist() -> dict:
    """Load the precomputed file event counts JSON from nEvents_scan."""
    base_path = Path("/afs/.cern.ch/work/p/phploner/foundation_model_testing/src/utils/nEvents_scan/file_event_counts.json")
    if not base_path.exists():
        raise FileNotFoundError(f"Global file list not found: {base_path}")
    with open(base_path) as f:
        data = json.load(f)
    print(f"🟢 Loaded global file list ({sum(len(v) for v in data.values())} files) from {base_path}")
    return data

def class_seed(folder: str, seed: int) -> int:
    """Seed for one class, derived from its folder name and the global seed.

    Each class draws from its own generator so that the files chosen for one
    class do not depend on which other classes are in the config: a single
    shared generator advances once per class, so adding or removing one shifts
    the draw of every class after it.

    `hashlib`, not the built-in `hash()`, because Python salts the hash of a
    string per process, which would make the manifest differ between runs.
    """
    digest = hashlib.blake2b(f"{seed}:{folder}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


MANIFEST_STRATEGIES = ("per_class", "legacy")


def make_split_manifest(
    global_filelist, split_counts, include_folders, seed=42, strategy="per_class"
):
    """
    Absolute target rows per class: greedily assign whole files until
    each split reaches its target in `split_counts` (within one file).
    Extra files are ignored once test target is met.

    `strategy` picks how the files of a class are drawn:

    "per_class" (default)
        One generator per class, seeded from the class name through
        `class_seed`, over the file names in sorted order. The files of a class
        then depend only on that class, the seed and the targets, so a class can
        be added to or removed from `to_classify` without changing the others.

    "legacy"
        What the repository did before: a single generator for every class, used
        in the order the classes appear in the config, over the file names in the
        order the event-count JSON happens to list them. Reproduces datasets
        built with that code; do not use it for new ones. Adding or removing a
        class here changes the files of every class drawn after it.
    """
    import numpy as np

    if strategy not in MANIFEST_STRATEGIES:
        raise ValueError(
            f"Unknown manifest strategy {strategy!r}; expected one of {MANIFEST_STRATEGIES}."
        )

    split_names = ["train", "val", "test"]
    targets_abs = np.array(split_counts, dtype=int)  # e.g., [50000, 20000, 20000]

    if strategy == "legacy":
        print("⚠️  Building the split manifest with the legacy shared generator: "
              "the files of one class depend on which other classes are selected.")
        shared_rng = np.random.default_rng(seed)

    manifest = {}
    for folder in include_folders:
        if strategy == "legacy":
            items = [(fn, int(n)) for fn, n in global_filelist.get(folder, {}).items()]
        else:
            # Sorted, so the order does not depend on how the event-count JSON
            # happens to be written; the shuffle below is what randomises it.
            items = sorted((fn, int(n)) for fn, n in global_filelist.get(folder, {}).items())

        if not items:
            manifest[folder] = {s: [] for s in split_names}
            continue

        if strategy == "legacy":
            shared_rng.shuffle(items)
        else:
            rng = np.random.default_rng(class_seed(folder, seed))
            items = [items[i] for i in rng.permutation(len(items))]

        buckets = {s: [] for s in split_names}
        split_idx = 0
        acc = 0

        for fname, n in items:
            # if all splits done, stop
            if split_idx >= len(split_names):
                break

            buckets[split_names[split_idx]].append(fname)
            acc += n

            # move to next split once target reached (allow overshoot by one file)
            if acc >= targets_abs[split_idx]:
                print(f"Folder {folder} added enough files (n={acc}) to split {split_names[split_idx]}")
                split_idx += 1
                acc = 0

        # ensure remaining splits exist as empty lists
        for s in split_names:
            buckets.setdefault(s, [])

        manifest[folder] = buckets

    return manifest


# ============================================================
# READING, WRITING AND CHECKING THE SPLIT MANIFEST
# ============================================================

MANIFEST_NAME = "split_manifest.json"
MANIFEST_META_KEY = "_meta"
SPLIT_NAMES = ("train", "val", "test")


def split_manifest_meta(seed: int, strategy: str, split_counts) -> dict:
    """What the manifest records about how it was built, so it can be checked."""
    return {
        "seed": int(seed),
        "manifest_strategy": str(strategy),
        "split_counts": [int(n) for n in split_counts],
    }


def save_split_manifest(path: str, manifest: dict, meta: dict) -> None:
    """Write the manifest with its metadata under the reserved `_meta` key."""
    with open(path, "w") as f:
        json.dump({MANIFEST_META_KEY: meta, **manifest}, f, indent=2)


def load_split_manifest(path: str) -> tuple[dict, dict | None]:
    """Return (folders, meta). `meta` is None for a manifest written before it existed."""
    with open(path) as f:
        payload = json.load(f)
    meta = payload.get(MANIFEST_META_KEY)
    folders = {k: v for k, v in payload.items() if k != MANIFEST_META_KEY}
    return folders, meta


def check_manifest_meta(meta: dict | None, expected: dict, path: str) -> None:
    """Refuse to extend a dataset whose manifest was built with other settings.

    Vectorized and preprocessed shards only mean something together with the
    manifest that chose them, so a run that would add files drawn differently
    stops here instead of leaving two selections mixed in one directory.
    """
    if meta is None:
        print(
            f"⚠️  {path} carries no metadata: it was written before the manifest "
            "recorded how it was built, so the seed and the strategy cannot be "
            "checked. It is used as it is."
        )
        return

    differences = [
        f"  {key}: manifest has {meta.get(key)!r}, this run asks for {value!r}"
        for key, value in expected.items()
        if meta.get(key) != value
    ]
    if differences:
        raise ValueError(
            f"The existing manifest {path} does not match this configuration:\n"
            + "\n".join(differences)
            + "\nThe shards next to it were chosen with the settings the manifest "
            "records. Either restore those settings, or give this dataset a new "
            "`data.label` so the new selection gets its own directory."
        )


def check_shards_match_manifest(eos_vec_dir: str, manifest: dict, strict: bool | None = None) -> None:
    """Every shard already on disk must be a file the manifest selected.

    Catches the case this is really here for: a dataset vectorized with one
    selection, then extended with another, which leaves a directory holding two
    samples with no error anywhere.

    `strict` defaults to whether the stored manifest records how it was built.
    Datasets produced before it did are only warned about: they predate the rule,
    and some of them do hold shards their manifest no longer lists.
    """
    if strict is None:
        stored = os.path.join(eos_vec_dir, MANIFEST_NAME)
        strict = True
        if os.path.exists(stored):
            _, meta = load_split_manifest(stored)
            strict = meta is not None

    for split in SPLIT_NAMES:
        for folder, buckets in manifest.items():
            split_dir = os.path.join(eos_vec_dir, split, folder)
            if not os.path.isdir(split_dir):
                continue
            selected = {fn.replace(".parquet", "_x.npy") for fn in buckets.get(split, [])}
            found = {f for f in os.listdir(split_dir) if f.endswith("_x.npy")}
            unexpected = sorted(found - selected)
            if not unexpected:
                continue

            shown = ", ".join(unexpected[:5])
            more = f" and {len(unexpected) - 5} more" if len(unexpected) > 5 else ""
            message = (
                f"{split_dir} holds {len(unexpected)} shard(s) the manifest does not "
                f"select: {shown}{more}.\nThey come from a different file selection. "
                "Vectorize into a new `data.label`, or remove them, rather than "
                "mixing two samples in one directory."
            )
            if strict:
                raise ValueError(message)
            print(f"⚠️  {message}")


def resolve_split_manifest(
    manifest_path: str,
    include_folders: list,
    split_counts,
    seed: int = 42,
    strategy: str = "per_class",
    global_filelist: dict | None = None,
) -> dict:
    """The one way a manifest is obtained: reuse the stored one, or build it.

    Reusing it is what keeps a dataset stable, so an existing manifest is never
    rewritten; it is only extended, and only with classes it does not yet cover,
    which `per_class` allows without touching the classes already there.
    """
    expected = split_manifest_meta(seed, strategy, split_counts)

    if os.path.exists(manifest_path):
        manifest, meta = load_split_manifest(manifest_path)
        print(f"🟡 Using existing split manifest: {manifest_path}")
        check_manifest_meta(meta, expected, manifest_path)
    else:
        manifest, meta = {}, None

    missing = [f for f in include_folders if f not in manifest]
    if missing:
        if manifest:
            if strategy != "per_class":
                raise ValueError(
                    f"{manifest_path} does not cover {missing}, and strategy "
                    f"{strategy!r} draws every class from one generator, so adding a "
                    "class would change the files of the others. Use a new "
                    "`data.label` for this selection."
                )
            print(f"🟢 Extending the manifest with {len(missing)} new class(es): {missing}")
        else:
            print(f"🟢 Building a new split manifest for {len(missing)} class(es) ...")
        for folder in missing:
            print(f"   • {folder}")

        if global_filelist is None:
            global_filelist = load_global_filelist()
        manifest.update(
            make_split_manifest(
                global_filelist=global_filelist,
                split_counts=split_counts,
                include_folders=missing,
                seed=seed,
                strategy=strategy,
            )
        )
        save_split_manifest(manifest_path, manifest, expected)
        print(f"✅ Wrote split manifest → {manifest_path}")

    return manifest


# ============================================================
# VECTORIZE AND SAVE LOCALLY
# ============================================================


def vectorize_to_local(
    base_dir: str,
    config: dict,
    class_names: list,
    folder_map: dict,
    labels_map: dict,
    all_cols: list,
    vlen: int,
    tmp_vec_dir: str,
    eos_vec_dir: str,
    split_counts: list,  # [train, val, test]
    read_batch_size: int = 512,
    split_manifest: dict | None = None,
    parallel_processing: bool = False,
    seed: int = 42,
    manifest_strategy: str = "per_class",
):
    """Vectorize Parquet shards using a deterministic split manifest.

    If `split_manifest_path` is given, it defines which files belong
    to train/val/test. Otherwise, the manifest is created or reused
    in `eos_vec_dir/split_manifest.json`, with `seed` and `manifest_strategy`
    deciding which files each class contributes (see `make_split_manifest`).

    Each file in the manifest is converted into .npy shards under:
        eos_vec_dir/{train,val,test}/{class_folder}/...
    """

    os.makedirs(tmp_vec_dir, exist_ok=True)
    os.makedirs(eos_vec_dir, exist_ok=True)
    save_feature_map(config, eos_vec_dir, vlen)

    # -------------------------------------------------------------------------
    # Load or create manifest
    # -------------------------------------------------------------------------
    if split_manifest is not None:
        # A subset handed over by a batch job: the full manifest, and the checks
        # against it, belong to the script that split the work.
        print("🟡 Using in-memory split manifest (dict provided).")
    else:
        split_manifest = resolve_split_manifest(
            manifest_path=os.path.join(eos_vec_dir, MANIFEST_NAME),
            include_folders=[folder_map[c] for c in class_names if c in folder_map],
            split_counts=split_counts,
            seed=seed,
            strategy=manifest_strategy,
        )
        check_shards_match_manifest(eos_vec_dir, split_manifest)

    # -------------------------------------------------------------------------
    # Vectorize files based on manifest
    # -------------------------------------------------------------------------
    split_names = ["train", "val", "test"]

    for cname in class_names:
        class_folder = folder_map[cname]
        label_id = labels_map[cname]
        if class_folder not in split_manifest:
            print(f"⚪ Skipping {cname}: not in manifest.")
            continue

        for split_name in split_names:
            file_list = split_manifest[class_folder].get(split_name, [])
            if not file_list:
                continue

            print(f"🟡 Processing {cname}/{split_name} ({len(file_list)} files)")

            for fname in file_list:
                folder = os.path.join(base_dir, class_folder)
                path = os.path.join(folder, fname)
                base = os.path.splitext(fname)[0]

                if parallel_processing:
                    scratch = os.environ.get("TMPDIR", "/tmp")
                    tmp_split_dir = os.path.join(scratch, split_name, class_folder)
                else:
                    tmp_split_dir = os.path.join(tmp_vec_dir, split_name, class_folder)

                eos_split_dir = os.path.join(eos_vec_dir, split_name, class_folder)
                os.makedirs(tmp_split_dir, exist_ok=True)
                os.makedirs(eos_split_dir, exist_ok=True)

                dst_x = os.path.join(eos_split_dir, f"{base}_x.npy")
                dst_y = os.path.join(eos_split_dir, f"{base}_y.npy")

                # Idempotent skip if already vectorized
                if os.path.exists(dst_x) and os.path.exists(dst_y):
                    continue

                print(f"→ Processing {path}")
                all_feats, all_labels = [], []

                try:
                    with pq.ParquetFile(path) as pqf:
                        for batch in pqf.iter_batches(columns=all_cols, batch_size=read_batch_size):
                            tbl = pa.Table.from_batches([batch])
                            arrays = {col: to_awkward_column(tbl[col]) for col in all_cols}
                            feats = build_vectors_batch(arrays, config, fill=0.0)
                            n = feats.shape[0]
                            all_feats.append(feats)
                            all_labels.append(np.full((n,), label_id, dtype=np.int64))
                except Exception as e:
                    print(f"❌ Error reading {path}: {e}")
                    continue

                if not all_feats:
                    continue

                feats_cat = np.concatenate(all_feats, axis=0)
                labels_cat = np.concatenate(all_labels, axis=0)

                local_x = os.path.join(tmp_split_dir, f"{base}_x.npy")
                local_y = os.path.join(tmp_split_dir, f"{base}_y.npy")
                np.save(local_x, feats_cat)
                np.save(local_y, labels_cat)

                move_to_eos(local_x, dst_x)
                move_to_eos(local_y, dst_y)

                os.remove(local_x) if os.path.exists(local_x) else None
                os.remove(local_y) if os.path.exists(local_y) else None

                print(f"✅ Saved {split_name}/{base}: {feats_cat.shape}")

    print(f"✅ Finished vectorizing → {eos_vec_dir}")


# ============================================================
# RESEEDING EVERY EPOCH
# ============================================================
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return  # not running inside a DataLoader worker
    # seed everything based on torch's worker-specific initial seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# CHECK THAT A TARGET DIRECTORY HOLDS THE EVENTS THAT WERE ASKED FOR
# ============================================================


def count_split_events(split_dir: str) -> tuple[int, int]:
    """(events, shards) in one split directory, read from the .npy headers.

    Reading the header costs one file open and no data, which is what the loader
    does too, and it describes the shards as they are rather than as a snapshot
    of the Parquet files says they should be.
    """
    shards = sorted(f for f in os.listdir(split_dir) if f.endswith("_x.npy"))
    events = sum(np.load(os.path.join(split_dir, f), mmap_mode="r").shape[0] for f in shards)
    return events, len(shards)


def has_enough_events(
    target: str,
    train_val_test_split_per_class,
    classnames,
    folder_map,
) -> bool:
    """True when every (split, class) holds at least the events asked for.

    The counts come from the shards themselves. They used to come from
    `file_event_counts.json`, which describes the Parquet files: that missed a
    shard written short, raised KeyError on any file the snapshot did not list,
    and could not be used at all for data downloaded from Hugging Face, whose
    files are not in the snapshot.

    Args:
        target: directory holding train/val/test/{folder}/
        train_val_test_split_per_class: e.g. [50_000, 20_000, 20_000]
        classnames: the classes to check, e.g. ["QCD_inclusive", "ggHbb"]
        folder_map: class name -> directory name
    """
    if not target or not os.path.exists(target):
        print(f"❌ {target} does not exist.")
        return False

    for split, needed in zip(SPLIT_NAMES, train_val_test_split_per_class):
        for cname in classnames:
            folder = folder_map[cname]
            split_dir = os.path.join(target, split, folder)

            if not os.path.isdir(split_dir):
                print(f"❌ Missing directory: {split_dir}")
                return False

            events, shards = count_split_events(split_dir)
            if shards == 0:
                print(f"❌ No *_x.npy shards in {split_dir}")
                return False

            if events < needed:
                print(
                    f"❌ {split}/{cname}: {events:,} events in {shards} shards, "
                    f"{needed:,} requested."
                )
                return False

            print(f"✅ {split}/{cname}: {events:,} events in {shards} shards (need {needed:,})")

    return True
