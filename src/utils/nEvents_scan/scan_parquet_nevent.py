#!/usr/bin/env python3
"""Record how many events each Parquet file holds, so planning splits is cheap.

The result is what `data.event_counts_json` points at:

    {
        "QCD_HT50toInf": {"QCD_HT50toInf-NEVENT10000-RS26000001.parquet": 8243, ...},
        "HH_4b": {"HH_4b-NEVENT10000-RS20000001.parquet": 9993, ...},
    }

Only the Parquet footer is read, so scanning is quick compared with reading the files.

    # the copy on EOS, every process the default config knows about
    python src/utils/nEvents_scan/scan_parquet_nevent.py

    # a subset downloaded from Hugging Face: every directory found there
    python src/utils/nEvents_scan/scan_parquet_nevent.py \
        --base-dir data/collide --output data/collide/file_event_counts.json

The shipped src/utils/nEvents_scan/file_event_counts.json covers the EOS production as
it was scanned; rerun this when the dataset changes or when a process is missing.
"""

import argparse
import json
import os
from omegaconf import OmegaConf
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def count_events(path: str) -> tuple[str, int | str]:
    """Return the number of events in a Parquet file."""
    try:
        pf = pq.ParquetFile(path)
        # Try standard num_rows first
        nrows = pf.metadata.num_rows
        if nrows is None:
            # Try custom metadata fallback (nEvents from your writer)
            meta = pf.metadata.metadata or {}
            if b"nEvents" in meta:
                nrows = int(meta[b"nEvents"])
            else:
                # Sum row groups if both missing (rare)
                nrows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
        return os.path.basename(path), nrows
    except Exception as e:
        return os.path.basename(path), f"error: {e}"


def scan_dataset(base_dir: str, process_to_folder: dict, output_json: str, n_workers: int = 8):
    """Iterate over all folders and scan parquet files in parallel."""
    summary = {}
    for process_name, folder_name in process_to_folder.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"⚠️  Missing folder: {folder_path}, skipping")
            continue

        files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path)
                 if f.endswith(".parquet")]
        print(f"🔍 Scanning {folder_name} ({len(files)} files)")

        class_meta = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for fname, nrows in tqdm(ex.map(count_events, files), total=len(files)):
                if isinstance(nrows, int):
                    class_meta[fname] = nrows
                else:
                    print(f"  ⚠️  {folder_name}/{fname}: {nrows}")

        summary[folder_name] = class_meta

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved {output_json} ({len(summary)} datasets)")


def folders_in(base_dir: str) -> dict:
    """Every directory holding Parquet files, as {name: name}.

    What a Hugging Face download looks like: directories named after the processes,
    and no config saying which ones are there.
    """
    found = {}
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and any(f.endswith(".parquet") for f in os.listdir(path)):
            found[name] = name
    return found


def main():
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "file_event_counts.json")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=None,
                    help="directory holding one folder per process "
                         "(default: paths.dataset_dir of configs/paths/fm_testing.yaml)")
    ap.add_argument("--output", default=default_output, help="where to write the JSON")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if args.base_dir:
        base_dir = args.base_dir
        process_to_folder = folders_in(base_dir)
        print(f"🔍 {len(process_to_folder)} process folders found under {base_dir}")
    else:
        # dataset_dir is ${oc.env:COLLIDE_DATASET_DIR,<the copy on EOS>}; reading the
        # node resolves it, and nothing else in that file is touched.
        paths = OmegaConf.load(os.path.join(repo, "configs/paths/fm_testing.yaml"))
        base_dir = str(paths.dataset_dir)
        data_cfg = OmegaConf.load(os.path.join(repo, "configs/data/collide2v_basic.yaml"))
        process_to_folder = dict(data_cfg["process_to_folder"])
        print(f"🔍 {len(process_to_folder)} processes from configs/data/collide2v_basic.yaml")

    scan_dataset(base_dir, process_to_folder, args.output, n_workers=args.workers)


if __name__ == "__main__":
    main()
