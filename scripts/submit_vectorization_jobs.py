#!/usr/bin/env python3
"""
submit_vectorization_jobs.py

Use this script to batch submit vectorization jobs to Condor.

Reads Hydra configs, resolves the dataset's split manifest, splits it into chunks of ≤ MAX_FILES_PER_JOB,
and directly submits each chunk to Condor to run `vectorize_to_local` inside the Apptainer container.
"""

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import os
import json
import subprocess
import sys
import hydra
from math import ceil
from pathlib import Path
from omegaconf import DictConfig

from src.data.utils import MANIFEST_NAME, check_shards_match_manifest, resolve_split_manifest

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
MAX_FILES_PER_JOB = 50
JOB_FLAVOUR = "tomorrow"  # ≈ 1 day runtime per job

PROJECT_DIR = Path(__file__).resolve().parents[1]
WRAPPER_SCRIPT = PROJECT_DIR / "src/data/wrapper_vectorize.sh"
LOG_DIR = PROJECT_DIR / "logs/condor_logs/vectorization"

def experiment_override() -> str:
    """The `experiment=` the user typed, so the jobs run the same one.

    Hydra has already consumed it into the composed config by the time this runs, and
    the composed config does not say which experiment file it came from, so it is read
    back from the command line. Empty means the jobs use the config default.
    """
    for arg in sys.argv[1:]:
        if arg.startswith("experiment="):
            return arg.split("=", 1)[1]
    return ""


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
@hydra.main(
    config_path="../configs",
    config_name="vectorize_preprocess.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):

    print("🟢 Hydra config composed successfully")
    # -----------------------------------------------------------------------------
    # LOAD CONFIGS
    # -----------------------------------------------------------------------------
    print("🟢 Loading Hydra configs...")

    data_cfg = cfg.data
    paths_cfg = cfg.paths

    LABEL = data_cfg.label
    TMP_VEC_DIR = Path(paths_cfg.tmp_data_dir) / LABEL / "vectorized"
    EOS_VEC_DIR = Path(paths_cfg.eos_data_dir) / LABEL / "vectorized"
    DATASET_DIR = Path(paths_cfg.dataset_dir)

    split_counts = data_cfg.train_val_test_split_per_class
    include_folders = [data_cfg.process_to_folder[c] for c in data_cfg.to_classify]

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------------
    # RESOLVE THE MANIFEST
    #
    # The same file the datamodule reads and writes, `split_manifest.json` next to
    # the shards, so the batch path and the local path cannot disagree about which
    # files the dataset is made of.
    # -----------------------------------------------------------------------------
    print("\n🟢 Resolving split manifest...")
    os.makedirs(EOS_VEC_DIR, exist_ok=True)

    manifest = resolve_split_manifest(
        manifest_path=str(EOS_VEC_DIR / MANIFEST_NAME),
        include_folders=include_folders,
        split_counts=split_counts,
        seed=data_cfg.get("seed", cfg.get("seed", 42)),
        strategy=data_cfg.get("manifest_strategy", "per_class"),
    )
    check_shards_match_manifest(str(EOS_VEC_DIR), manifest)

    # -----------------------------------------------------------------------------
    # FLATTEN MANIFEST INTO ENTRIES
    # -----------------------------------------------------------------------------
    entries = []
    for folder in include_folders:
        for split_name, files in manifest[folder].items():
            for fname in files:
                target_path = EOS_VEC_DIR / split_name / folder / fname.replace(".parquet", "_x.npy")
                if not target_path.exists():
                    entries.append((folder, split_name, fname))

    n_jobs = ceil(len(entries) / MAX_FILES_PER_JOB)
    print(f"\n🟡 Total files: {len(entries)} → {n_jobs} jobs (≤{MAX_FILES_PER_JOB} files/job)")

    # -----------------------------------------------------------------------------
    # SUBMIT ALL JOBS
    # -----------------------------------------------------------------------------
    for i in range(n_jobs):
        start = i * MAX_FILES_PER_JOB
        end = start + MAX_FILES_PER_JOB
        chunk = entries[start:end]
        submit_job(i, chunk, experiment_override())

    print(f"\n✅ All {n_jobs} jobs submitted to Condor.")

# -----------------------------------------------------------------------------
# FUNCTION TO SUBMIT A SINGLE JOB
# -----------------------------------------------------------------------------
def submit_job(job_idx, chunk, experiment=""):
    # Build submanifest dict
    submanifest = {}
    for folder, split_name, fname in chunk:
        submanifest.setdefault(folder, {"train": [], "val": [], "test": []})
        submanifest[folder][split_name].append(fname)

    # Save submanifest to a temporary JSON file
    manifest_path = LOG_DIR / f"manifest_{job_idx:04d}.json"
    with open(manifest_path, "w") as f:
        json.dump(submanifest, f, indent=2)

    log_out = LOG_DIR / f"job_{job_idx:04d}.out"
    log_err = LOG_DIR / f"job_{job_idx:04d}.err"
    log_log = LOG_DIR / f"job_{job_idx:04d}.log"

    # Run the wrapper with the manifest path as argument
    submit_content = f"""\
executable = {WRAPPER_SCRIPT}
arguments  = {manifest_path} {PROJECT_DIR} {experiment}
initialdir = {LOG_DIR}

output = {log_out}
error  = {log_err}
log    = {log_log}

# No stream_output/stream_error: the CERN schedd refuses the whole submission
# ("stream_out and stream_err are no longer supported"), and it refuses it at commit
# time, so condor_submit -dry-run accepts the file and only a real submit fails.
# The logs above are written when the job ends.

run_as_owner = True
+JobFlavour = "{JOB_FLAVOUR}"

# getenv = False: the job gets the environment set here and nothing else. Shipping
# the submitting shell's environment makes a job depend on the terminal it was sent
# from — a stray PYTHONPATH or CONDA_PREFIX is enough to change what runs — and the
# wrapper calls apptainer with --cleanenv anyway. apptainer lives in /usr/bin, which
# the default PATH covers.
getenv = False
request_cpus = 4
environment = "OMP_NUM_THREADS=4; MKL_NUM_THREADS=4"

queue
"""

    sub_path = LOG_DIR / f"vectorize_job_{job_idx:04d}.sub"
    with open(sub_path, "w") as f:
        f.write(submit_content)

# check=True: a rejected submission used to be reported by condor_submit and then
    # ignored here, so a loop could print "submitted" for hundreds of jobs that never
    # existed.
    subprocess.run(["condor_submit", str(sub_path)], check=True)
    print(f"🚀 Submitted job {job_idx:04d} ({len(chunk)} files)")

if __name__ == "__main__":
    main()
