#!/usr/bin/env python3
"""Download a few files of COLLIDE-1M from Hugging Face, and scan their event counts.

The public copy, https://huggingface.co/datasets/fastmachinelearning/collide-1m, is
about 953 GB over 740 files, so nobody downloads it whole: this takes the first files
of the processes you name, which is enough to run the whole pipeline.

    # two files each of QCD and HH->4b, about 6 GB
    python scripts/download_collide_subset.py --out data/collide \\
        --processes QCD_HT50toInf HH_4b --files-per-process 2

    # see what is there without downloading
    python scripts/download_collide_subset.py --list

It then writes `<out>/file_event_counts.json`, because a manifest is planned from event
counts and the snapshot in the repository only covers the copy on EOS. Point the
pipeline at both:

    python src/prepare_data.py experiment=fm_testing_18class_highlevel \\
        paths.dataset_dir=data/collide \\
        data.event_counts_json=data/collide/file_event_counts.json \\
        data.label=hf_subset

Use the default feature set, `collide2v_common`: the Hugging Face copy does not have
the nine variables `collide2v_extended_eos` adds, and it is NOT the same events as the
copy on EOS — see section 3 of the README before comparing any number.
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ID = "fastmachinelearning/collide-1m"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/collide", help="where to put the files")
    ap.add_argument("--processes", nargs="*", default=["QCD_HT50toInf", "HH_4b"],
                    help="folder names, as on EOS (default: QCD_HT50toInf HH_4b)")
    ap.add_argument("--files-per-process", type=int, default=2)
    ap.add_argument("--list", action="store_true",
                    help="print how many files each process has, then stop")
    ap.add_argument("--no-scan", action="store_true",
                    help="skip writing file_event_counts.json")
    args = ap.parse_args()

    try:
        from huggingface_hub import list_repo_files, snapshot_download
    except ImportError:
        print("huggingface_hub is missing: pip install huggingface_hub", file=sys.stderr)
        return 1

    available = [f for f in list_repo_files(REPO_ID, repo_type="dataset")
                 if f.endswith(".parquet")]
    by_process = {}
    for f in available:
        by_process.setdefault(f.split("/", 1)[0], []).append(f)

    if args.list:
        print(f"{len(available)} files over {len(by_process)} processes:")
        for name, files in sorted(by_process.items()):
            print(f"  {name:48s} {len(files):3d} files")
        return 0

    unknown = [p for p in args.processes if p not in by_process]
    if unknown:
        print(f"not in the dataset: {unknown}\nrun with --list to see the names",
              file=sys.stderr)
        return 1

    wanted = []
    for name in args.processes:
        files = sorted(by_process[name])[: args.files_per_process]
        print(f"  {name}: taking {len(files)} of {len(by_process[name])} files")
        wanted += files

    # Files are 1.2 to 1.6 GB each, so say what this will cost before it starts.
    print(f"\n⬇  {len(wanted)} files, roughly {len(wanted) * 1.4:.0f} GB, into {args.out}")
    snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=wanted,
                      local_dir=args.out)

    if args.no_scan:
        return 0

    counts = os.path.join(args.out, "file_event_counts.json")
    scan = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "src", "utils", "nEvents_scan", "scan_parquet_nevent.py")
    print(f"\n🔍 Scanning the downloaded files → {counts}")
    subprocess.run([sys.executable, scan, "--base-dir", args.out, "--output", counts],
                   check=True)

    with open(counts) as f:
        scanned = json.load(f)
    total = sum(sum(v.values()) for v in scanned.values())
    print(f"\n✅ {len(scanned)} processes, {total:,} events. Use with:\n"
          f"   paths.dataset_dir={args.out} data.event_counts_json={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
