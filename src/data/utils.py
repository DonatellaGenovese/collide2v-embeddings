import hashlib
import json
import os
import random
import shutil
import subprocess

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

DEFAULT_EVENT_COUNTS_JSON = (
    Path(__file__).resolve().parents[1] / "utils" / "nEvents_scan" / "file_event_counts.json"
)


def load_global_filelist(path=None) -> dict:
    """{folder: {parquet file: events}}, used to plan the splits.

    Defaults to the copy shipped with the repository, found relative to this file,
    so it works from any checkout and any working directory. It used to be an
    absolute path into one person's AFS work area, which no one else could read.

    `path` — `data.event_counts_json` in the config — points somewhere else, which
    is what a dataset the shipped snapshot does not cover needs; regenerate one with
    `src/utils/nEvents_scan/scan_parquet_nevent.py`.
    """
    base_path = Path(path) if path else DEFAULT_EVENT_COUNTS_JSON
    if not base_path.exists():
        raise FileNotFoundError(
            f"Event-count file not found: {base_path}\n"
            "Point `data.event_counts_json` at one, or regenerate it with "
            "src/utils/nEvents_scan/scan_parquet_nevent.py."
        )
    with open(base_path) as f:
        data = json.load(f)
    print(f"🟢 Loaded event counts for {sum(len(v) for v in data.values())} files from {base_path}")
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
    event_counts_json=None,
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
            global_filelist = load_global_filelist(event_counts_json)
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


# ============================================================
# WILL THE DATASET FIT WHERE IT IS ABOUT TO BE WRITTEN?
# ============================================================

# Vectorised float32 shards plus their preprocessed copies, which are wider because
# phi becomes (sin, cos) and categories become one-hot: about 1.25 to 1.3 times on the
# feature sets shipped. 2.5 times the raw size covers both with some room.
SPACE_FACTOR = 2.5
EVENTS_PER_FILE = 10_000  # the NEVENT10000 in the file names, an upper bound


def parse_fs_listquota(text: str) -> int | None:
    """Free bytes from the output of `fs listquota`, or None if it says no limit.

        Volume Name                    Quota       Used %Used   Partition
        user.dgenoves               10485760    6608517   63%         20%
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    fields = lines[1].split()
    if len(fields) < 3 or not fields[1].isdigit() or not fields[2].isdigit():
        return None
    quota_kb, used_kb = int(fields[1]), int(fields[2])
    return max(quota_kb - used_kb, 0) * 1024


def free_bytes(path: str) -> int:
    """Space left where `path` is, honest about AFS quotas.

    On AFS the filesystem reports the whole partition — 2.2 TB free, measured, on a home
    directory whose quota had 3.9 GB left — so the quota is asked of AFS itself with
    `fs listquota`. Anywhere else the filesystem's own figure is the right one.
    """
    target = os.path.abspath(path)
    while not os.path.exists(target):
        target = os.path.dirname(target)

    if os.path.realpath(target).startswith("/afs/"):
        try:
            out = subprocess.run(["fs", "listquota", target], capture_output=True,
                                 text=True, timeout=30)
            free = parse_fs_listquota(out.stdout)
            if free is not None:
                return free
        except (OSError, subprocess.TimeoutExpired):
            pass
    return shutil.disk_usage(target).free


def check_output_space(out_dir: str, n_files: int, vlen: int, free: int | None = None) -> None:
    """Stop before writing a dataset that will not fit.

    The default output directory is inside the repository, which on lxplus usually means
    an AFS home with a 10 GB quota, and a real dataset is hundreds of gigabytes. Running
    out halfway leaves a quota full, shards cut short, and jobs failing for a reason that
    has nothing to do with them; this estimates the size first and says where to write
    instead.
    """
    if n_files == 0 or vlen <= 0:
        return
    estimate = n_files * EVENTS_PER_FILE * vlen * 4 * SPACE_FACTOR
    available = free_bytes(out_dir) if free is None else free
    if estimate <= available:
        return
    raise RuntimeError(
        f"This dataset needs about {estimate / 1e9:.1f} GB — {n_files} Parquet files, "
        f"{vlen} features per event, vectorised and preprocessed — and {out_dir} has "
        f"{available / 1e9:.1f} GB left"
        + (" of its AFS quota" if os.path.realpath(out_dir).startswith("/afs/") else "")
        + ".\nWrite it somewhere with room: set paths.eos_data_dir, for instance in "
        "configs/local/default.yaml, to a directory in your EOS area — section 2 of the "
        "README gives one."
    )


def filter_empty_events(X: np.ndarray, feature_map: dict) -> np.ndarray:
    """Mask of the events that have at least one reconstructed object.

    An event is empty when every object slot — jets, electrons, muons, photons — has
    PT == 0. Only meaningful on raw vectorized data, where 0 still means padding.
    """
    keep = np.zeros(len(X), dtype=bool)
    for cfg in feature_map.values():
        topk = cfg.get("topk")
        if topk is None:
            continue  # a scalar group such as MET has no object slots
        start = cfg["start"]
        n_cols = len(cfg["columns"])
        pt_cols = [start + i * n_cols for i in range(topk)]
        keep |= (X[:, pt_cols] != 0.0).any(axis=1)
    return keep


def check_columns_present(base_dir: str, manifest: dict, all_cols: list) -> None:
    """Every requested column must exist in the files, checked before any work starts.

    One Parquet file per class is opened and only its schema read. A missing column used
    to surface as a pyarrow error inside a batch job, after fifty files had been read,
    or — on the Hugging Face copy, which has 174 of the 271 columns EOS has — for every
    job at once.
    """
    missing_by_folder = {}
    for folder, buckets in manifest.items():
        names = [fn for split in SPLIT_NAMES for fn in buckets.get(split, [])]
        if not names:
            continue
        path = os.path.join(base_dir, folder, sorted(names)[0])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is in the manifest but not on disk.")
        present = set(pq.ParquetFile(path).schema_arrow.names)
        missing = [c for c in all_cols if c not in present]
        if missing:
            missing_by_folder[folder] = (os.path.basename(path), missing)

    if not missing_by_folder:
        return

    lines = [f"  {folder} ({fname}): {', '.join(cols)}"
             for folder, (fname, cols) in sorted(missing_by_folder.items())]
    raise ValueError(
        "Columns the configuration asks for are not in the data:\n" + "\n".join(lines)
        + "\nThe copy on EOS has columns the one on Hugging Face does not. If these are "
        "the nine EOS-only variables, use `data: collide2v_common`, which keeps only "
        "what both copies have."
    )


def check_feature_map_matches(eos_vec_dir: str, config) -> None:
    """Refuse to add shards whose columns differ from the ones already there.

    The manifest records which files a dataset is made of, not which features were taken
    from them, so changing `datasets_config` and keeping the `label` used to append
    shards of a different width — or the same width with different columns, which is
    worse, because nothing would ever complain.
    """
    stored_path = os.path.join(eos_vec_dir, "feature_map.json")
    if not os.path.exists(stored_path):
        return
    with open(stored_path) as f:
        stored = json.load(f)
    wanted = build_feature_map_dict(config)
    if stored == wanted:
        return

    differences = []
    for group in sorted(set(stored) | set(wanted)):
        if group not in stored:
            differences.append(f"  {group}: not in the dataset, asked for now")
        elif group not in wanted:
            differences.append(f"  {group}: in the dataset, not asked for now")
        elif stored[group] != wanted[group]:
            differences.append(
                f"  {group}: dataset has topk={stored[group]['topk']} "
                f"count={stored[group]['count']} with {len(stored[group]['columns'])} columns, "
                f"this run asks for topk={wanted[group]['topk']} "
                f"count={wanted[group]['count']} with {len(wanted[group]['columns'])} columns"
            )
    raise ValueError(
        f"{stored_path} describes different features from the ones this run asks for:\n"
        + "\n".join(differences)
        + "\nShards already written cannot be read with this feature map. Give this "
        "selection a new `data.label`."
    )


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
    event_counts_json=None,
    drop_empty_events: bool = False,
    skip_unreadable_files: bool = False,
):
    """Vectorize Parquet shards using a deterministic split manifest.

    If `split_manifest_path` is given, it defines which files belong
    to train/val/test. Otherwise, the manifest is created or reused
    in `eos_vec_dir/split_manifest.json`, with `seed` and `manifest_strategy`
    deciding which files each class contributes (see `make_split_manifest`).

    Each file in the manifest is converted into .npy shards under:
        eos_vec_dir/{train,val,test}/{class_folder}/...

    `drop_empty_events` discards events in which every jet, electron, muon and photon
    slot has PT == 0. It is a physics choice, not a technicality, and it is off by
    default so that a dataset contains what the generator produced.

    Which events it touches is very uneven: measured over the vectorisation logs of
    the published study, about 5% of QCD_HT50toInf and under 0.5% of everything else.
    QCD is generated with HT > 50 GeV, so a real fraction of it has nothing above the
    storage thresholds — and QCD is also the class an anomaly-detection model is
    trained on as "normal", so the filter decides the softest edge of what normal
    means. Turn it on deliberately, and say so when reporting results.

    A Parquet file that cannot be read stops the run, but only after every other file
    has been done, so one bad file does not throw away a batch job's work: the error
    lists the files, and a batch job ends with a non-zero status. It used to be printed
    and skipped, which left a split short while every job reported success.
    `skip_unreadable_files` restores the skipping, for a known-bad file you choose to
    live without; the event check before training still reports any split it leaves
    short.

    Keeping them is safe for a classifier: the transformer applies no padding mask, so
    an all-zero event still yields well-defined tokens from the projection bias and the
    type embedding. It is not neutral for a contrastive objective, where every empty
    event maps to the same point and both augmented views of it are identical.
    """

    os.makedirs(tmp_vec_dir, exist_ok=True)
    os.makedirs(eos_vec_dir, exist_ok=True)
    check_feature_map_matches(eos_vec_dir, config)
    save_feature_map(config, eos_vec_dir, vlen)
    _filter_fm = build_feature_map_dict(config)

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
            event_counts_json=event_counts_json,
        )
        check_shards_match_manifest(eos_vec_dir, split_manifest)
        still_to_write = sum(
            not os.path.exists(os.path.join(eos_vec_dir, split, folder,
                                            fn.replace(".parquet", "_x.npy")))
            for folder, buckets in split_manifest.items()
            for split, files in buckets.items()
            for fn in files
        )
        check_output_space(eos_vec_dir, still_to_write, compute_vlen(config))

    check_columns_present(base_dir, split_manifest, all_cols)

    # -------------------------------------------------------------------------
    # Vectorize files based on manifest
    # -------------------------------------------------------------------------
    split_names = ["train", "val", "test"]
    unreadable = []

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
                    print(f"❌ Error reading {path}: {type(e).__name__}: {e}")
                    unreadable.append((path, f"{type(e).__name__}: {e}"))
                    continue

                if not all_feats:
                    continue

                feats_cat = np.concatenate(all_feats, axis=0)
                labels_cat = np.concatenate(all_labels, axis=0)

                if drop_empty_events:
                    keep = filter_empty_events(feats_cat, _filter_fm)
                    removed = int((~keep).sum())
                    if removed:
                        print(f"  ⚠️  Dropped {removed}/{len(feats_cat)} events with no "
                              f"reconstructed objects ({removed / len(feats_cat) * 100:.1f}%)")
                    feats_cat, labels_cat = feats_cat[keep], labels_cat[keep]

                local_x = os.path.join(tmp_split_dir, f"{base}_x.npy")
                local_y = os.path.join(tmp_split_dir, f"{base}_y.npy")
                np.save(local_x, feats_cat)
                np.save(local_y, labels_cat)

                move_to_eos(local_x, dst_x)
                move_to_eos(local_y, dst_y)

                os.remove(local_x) if os.path.exists(local_x) else None
                os.remove(local_y) if os.path.exists(local_y) else None

                print(f"✅ Saved {split_name}/{base}: {feats_cat.shape}")

    if unreadable:
        listing = "\n".join(f"  {path}\n      {error}" for path, error in unreadable)
        message = (
            f"{len(unreadable)} Parquet file(s) could not be read, so their splits are "
            f"short by those events:\n{listing}"
        )
        if not skip_unreadable_files:
            raise RuntimeError(
                message + "\nEvery other file was vectorised, and rerunning redoes only "
                "these. If a file is genuinely broken, set data.skip_unreadable_files=true "
                "to go on without it."
            )
        print(f"⚠️  {message}\nSkipped, because data.skip_unreadable_files is set.")

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


SHARD_ROWS_CACHE = ".shard_rows.json"


def shard_rows(split_dir: str) -> list[tuple[str, int]]:
    """(shard name, events) for every *_x.npy in a directory, sorted by name.

    The number of events is in the .npy header, and reading it means opening the file:
    74 ms a file on EOS, measured, so a twelve-class dataset took a minute and a half
    to check before training and as long again to plan the loader. A stat, by
    contrast, comes back with the directory listing and costs a fraction of a
    millisecond.

    So the counts are cached in `.shard_rows.json` next to the shards, keyed by size
    and modification time, and a header is read only for a shard that is new or has
    changed since. The cache is written when the directory is writable and ignored
    when it is not, which is the case for a dataset someone else owns.
    """
    cache_path = os.path.join(split_dir, SHARD_ROWS_CACHE)
    try:
        with open(cache_path) as f:
            cached = json.load(f)
    except (OSError, ValueError):
        cached = {}

    result, fresh, changed = [], {}, False
    for entry in sorted(os.scandir(split_dir), key=lambda e: e.name):
        if not entry.name.endswith("_x.npy"):
            continue
        st = entry.stat()
        signature = [st.st_size, st.st_mtime_ns]
        hit = cached.get(entry.name)
        if hit is not None and hit[:2] == signature:
            rows = hit[2]
        else:
            rows = int(np.load(entry.path, mmap_mode="r").shape[0])
            changed = True
        fresh[entry.name] = signature + [rows]
        result.append((entry.name, rows))

    if changed or set(fresh) != set(cached):
        # Write to a private name, then rename over the cache: several readers — DDP
        # ranks, parallel jobs — may do this at once, and each writes the same content.
        tmp = f"{cache_path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(fresh, f)
            os.replace(tmp, cache_path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    return result


def count_split_events(split_dir: str) -> tuple[int, int]:
    """(events, shards) in one split directory, from the shards themselves.

    The counts describe the shards as they are rather than as a snapshot of the
    Parquet files says they should be; `shard_rows` reads them without opening every
    file each time.
    """
    shards = shard_rows(split_dir)
    return sum(rows for _, rows in shards), len(shards)


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
