#!/usr/bin/env python3
"""
submit_preprocessing_jobs.py

Use this script to batch submit preprocessing jobs to Condor.

Reads Hydra configs, creates stats file, splits it into chunks of ≤ MAX_FILES_PER_JOB,
and directly submits each chunk to Condor to run `PreprocessingPipeline` inside the Apptainer container.
"""

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import json
import subprocess
import sys
import hydra
from math import ceil
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from src.preprocessing.preprocess import PreprocessingPipeline

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
MAX_FILES_PER_JOB = 50
JOB_FLAVOUR = "tomorrow"  # ~24h jobs


PROJECT_DIR = Path(__file__).resolve().parents[1]
WRAPPER_SCRIPT = PROJECT_DIR / "src/preprocessing/wrapper_preprocess.sh"
LOG_DIR = PROJECT_DIR / "logs/condor_logs/preprocessing"

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

    paths_cfg = cfg.paths
    data_cfg = cfg.data
    pre_cfg = cfg.preprocess
    class_order = data_cfg.to_classify              # {"QCD": "QCD inclusive", ...}
    process_to_folder = data_cfg.process_to_folder     # {"QCD inclusive": "QCD_HT50toInf", ...}

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------------
    # Build PreprocessingPipeline (fit_only)
    # -----------------------------------------------------------------------------
    print("🟡 Preparing preprocessing pipeline (fit_only)...")

    paths = {
        "eos_vec_dir": data_cfg.paths.eos_vec_dir,
        "tmp_preproc_dir": data_cfg.paths.tmp_preproc_dir,
        "eos_preproc_dir": data_cfg.paths.eos_preproc_dir,
    }

    stats_path = Path(paths["eos_preproc_dir"]) / "norm_stats.json"

    if not stats_path.exists():
        print("ℹ️ norm_stats.json not found, will fit normalization stats...")
        # Force mode = fit_only without touching config files
        pre_cfg_local = OmegaConf.to_container(pre_cfg, resolve=True)
        pre_cfg_local["mode"] = "fit_only"
        pre_cfg_local["enabled"] = True
        pre_cfg_local = OmegaConf.create(pre_cfg_local)

        pipeline = PreprocessingPipeline(
            paths=paths,
            preprocess_cfg=pre_cfg_local,
            process_to_folder=process_to_folder,
            class_order=class_order,
            device="cpu",
        )

        pipeline.run()
    else:
        print("ℹ️ norm_stats.json found, skipping fitting step.")

    if not stats_path.exists():
        raise RuntimeError(f"❌ norm_stats.json missing after fit_only: {stats_path}")

    print(f"✅ Fitted normalization stats → {stats_path}")

    # -----------------------------------------------------------------------------
    # SCAN VECTORIZED FILES
    # -----------------------------------------------------------------------------
    print("\n🟢 Scanning vectorized files...")

    entries = []  # list of (folder, split, filename)

    for split in ("train", "val", "test"):
        for cls in class_order:
            cls_folder = process_to_folder[cls]
            vec_dir = Path(paths["eos_vec_dir"]) / split / cls_folder
            pre_dir = Path(paths["eos_preproc_dir"]) / split / cls_folder
            npy_files = list(vec_dir.glob("*_x.npy"))
            target_npy_files = list(pre_dir.glob("*_x.npy"))
            target_npy_file_names = {f.name for f in target_npy_files}
            for f in npy_files:
                if f.name not in target_npy_file_names:
                    entries.append((cls_folder, split, f.name))

    num_files = len(entries)
    n_jobs = ceil(num_files / MAX_FILES_PER_JOB)

    print(f"🟡 Found {num_files} vectorized files → {n_jobs} jobs (≤{MAX_FILES_PER_JOB}/job)")

    # -----------------------------------------------------------------------------
    # SUBMIT ALL JOBS
    # -----------------------------------------------------------------------------
    for i in range(n_jobs):
        start = i * MAX_FILES_PER_JOB
        end = start + MAX_FILES_PER_JOB
        chunk = entries[start:end]
        submit_job(i, chunk, experiment_override())

    print("\n✅ All preprocessing jobs submitted.")


# -----------------------------------------------------------------------------
# FUNCTION TO SUBMIT A SINGLE JOB
# -----------------------------------------------------------------------------
def submit_job(job_idx, chunk, experiment=""):
    # Build submanifest structure
    submanifest = {}
    for folder, split, fname in chunk:
        if folder not in submanifest:
            submanifest[folder] = {"train": [], "val": [], "test": []}
        submanifest[folder][split].append(fname)

    # Save submanifest
    manifest_path = LOG_DIR / f"manifest_{job_idx:04d}.json"
    with open(manifest_path, "w") as f:
        json.dump(submanifest, f, indent=2)

    log_out = LOG_DIR / f"job_{job_idx:04d}.out"
    log_err = LOG_DIR / f"job_{job_idx:04d}.err"
    log_log = LOG_DIR / f"job_{job_idx:04d}.log"

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

    sub_path = LOG_DIR / f"preprocess_job_{job_idx:04d}.sub"
    with open(sub_path, "w") as f:
        f.write(submit_content)

# check=True: a rejected submission used to be reported by condor_submit and then
    # ignored here, so a loop could print "submitted" for hundreds of jobs that never
    # existed.
    subprocess.run(["condor_submit", str(sub_path)], check=True)
    print(f"🚀 Submitted job {job_idx:04d} ({len(chunk)} files)")

if __name__ == "__main__":
    main()
