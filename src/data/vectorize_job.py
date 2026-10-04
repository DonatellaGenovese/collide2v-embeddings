import argparse
import json
import hydra
import sys
import os
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.utils import vectorize_to_local, get_all_cols

"""
Vectorize job script for vectorizing data based on a given manifest file.
"""

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("🟢 vectorize_job.py starting...", flush=True)

# Parse manifest path before Hydra gets control
parser = argparse.ArgumentParser()
parser.add_argument("--manifest-path", type=str, required=True)
args, _ = parser.parse_known_args()

# Make manifest path absolute now (before Hydra may change cwd)
args.manifest_path = os.path.abspath(args.manifest_path)

# Hide --manifest-path from Hydra, but keep everything else: clearing sys.argv
# entirely meant `experiment=...` on the command line was silently ignored and the
# job ran with whatever experiment the config happened to default to.
filtered = [sys.argv[0]]
i = 1
while i < len(sys.argv):
    if sys.argv[i] == "--manifest-path":
        i += 2  # the flag and its value
    elif sys.argv[i].startswith("--manifest-path="):
        i += 1
    else:
        filtered.append(sys.argv[i])
        i += 1
sys.argv = filtered

@hydra.main(version_base="1.3", config_path="../../configs", config_name="vectorize_preprocess.yaml")
def main(cfg: DictConfig):
    with open(args.manifest_path, "r") as f:
        split_manifest = json.load(f)

    data_cfg = cfg.data
    to_classify = data_cfg.to_classify
    folder_map = data_cfg.process_to_folder

    vectorize_to_local(
        base_dir=data_cfg.paths.dataset_dir,
        config=data_cfg.datasets_config,
        class_names=to_classify,
        folder_map=folder_map,
        labels_map={k: i for i, k in enumerate(to_classify)},
        all_cols=get_all_cols(data_cfg.datasets_config),
        vlen=getattr(data_cfg, "vlen", 0),
        tmp_vec_dir=data_cfg.paths.tmp_vec_dir,
        eos_vec_dir=data_cfg.paths.eos_vec_dir,
        split_counts=data_cfg.train_val_test_split_per_class,
        split_manifest=split_manifest,
        parallel_processing=True,
        # Everything that changes what is written has to reach the job, or the batch
        # route and src/prepare_data.py build different datasets from one config.
        # drop_empty_events did not, so batch jobs always kept empty events.
        drop_empty_events=data_cfg.get("drop_empty_events", False),
        skip_unreadable_files=data_cfg.get("skip_unreadable_files", False),
    )

if __name__ == "__main__":
    main()
