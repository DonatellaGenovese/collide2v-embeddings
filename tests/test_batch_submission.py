"""Tests for what a batch job is given to find the container.

The jobs run with getenv = False, so they see only the variables their submit file
lists, and the wrappers find the .sif through FM_TESTING_IMAGE. That variable was not
among them: a job fell back to fm_testing.sif inside the repository, and with the image
on EOS, where the README tells students to build it, every batch job failed.
"""

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SUBMITTERS = ("scripts/submit_vectorization_jobs.py", "scripts/submit_preprocessing_jobs.py")
SUB_FILES = ("src/train_full_pipeline.sub", "src/eval.sub", "src/probe.sub")


def load(path):
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", SUBMITTERS)
def test_the_image_reaches_the_job_when_set(path, monkeypatch, tmp_path):
    monkeypatch.setenv("FM_TESTING_IMAGE", "fm_testing.sif")
    monkeypatch.chdir(tmp_path)
    env = load(path).job_environment()
    # Absolute, since the job does not start where the submission was made.
    assert f"FM_TESTING_IMAGE={tmp_path / 'fm_testing.sif'}" in env.split()
    assert "OMP_NUM_THREADS=4" in env.split()


@pytest.mark.parametrize("path", SUBMITTERS)
def test_without_it_the_job_gets_no_image_at_all(path, monkeypatch):
    monkeypatch.delenv("FM_TESTING_IMAGE", raising=False)
    assert "FM_TESTING_IMAGE" not in load(path).job_environment()


@pytest.mark.parametrize("path", SUBMITTERS)
def test_the_submit_file_is_written_with_that_environment(path):
    assert 'environment = "{job_environment()}"' in (REPO / path).read_text()


@pytest.mark.parametrize("path", SUB_FILES)
def test_the_static_submit_files_forward_it_with_an_empty_default(path):
    """`$ENV(X)` alone gives the job the string UNDEFINED when X is unset, which the
    wrapper would take for an image path; the empty default makes it fall back instead."""
    environment = [line for line in (REPO / path).read_text().splitlines()
                   if line.startswith("environment")]
    assert len(environment) == 1
    assert "FM_TESTING_IMAGE=$ENV(FM_TESTING_IMAGE:)" in environment[0]


# ---------------------------------------------------------------------------
# Training and evaluation jobs
#
# The training wrapper ran one fixed Optuna sweep whatever was asked for, and decided
# on the GPU by running hydra outside the container, where the batch nodes have none;
# the evaluation job ran eval.py without a checkpoint, which it requires. Neither
# wrapper was executable in Git.
# ---------------------------------------------------------------------------

WRAPPERS = {"src/wrapper_full_pipeline.sh": "src/train.py", "src/wrapper_eval.sh": "src/eval.py"}


def run_wrapper(path, args, tmp_path, gpu=False):
    """Run a wrapper with a fake apptainer that prints what it was asked to run."""
    import os
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "apptainer").write_text('#!/bin/bash\necho "APPTAINER $*"\n')
    (bin_dir / "apptainer").chmod(0o755)
    if gpu:
        (bin_dir / "nvidia-smi").write_text("#!/bin/bash\necho 'GPU 0: fake'\n")
        (bin_dir / "nvidia-smi").chmod(0o755)
    image = tmp_path / "fm.sif"
    image.write_text("")
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "FM_TESTING_IMAGE": str(image),
           "HOME": str(tmp_path)}
    return subprocess.run(["bash", str(REPO / path), *args], env=env, cwd=tmp_path,
                          capture_output=True, text=True)


@pytest.mark.parametrize("path", WRAPPERS)
def test_the_wrappers_can_be_executed(path):
    import os

    assert os.access(REPO / path, os.X_OK), f"{path} is not executable"


@pytest.mark.parametrize("path", WRAPPERS)
def test_no_python_runs_outside_the_container(path):
    for line in (REPO / path).read_text().splitlines():
        code = line.split("#", 1)[0]
        if "python" in code:
            assert '"${IMAGE}" python' in code, f"python outside the container: {line}"


def test_training_runs_the_experiment_it_was_given(tmp_path):
    result = run_wrapper("src/wrapper_full_pipeline.sh",
                         ["fm_testing_binary", "trainer=cpu", "trainer.max_epochs=1"], tmp_path)
    assert result.returncode == 0, result.stderr
    call = [line for line in result.stdout.splitlines() if line.startswith("APPTAINER")]
    assert call == [f"APPTAINER exec --bind /afs:/afs --bind /eos:/eos --writable-tmpfs {tmp_path}/fm.sif "
                    "python src/train.py experiment=fm_testing_binary trainer=cpu trainer.max_epochs=1"]


def test_the_gpu_is_used_when_the_node_has_one(tmp_path):
    result = run_wrapper("src/wrapper_full_pipeline.sh", ["fm_testing_binary"], tmp_path, gpu=True)
    assert "APPTAINER exec --nv " in result.stdout


def test_evaluation_is_given_its_checkpoint(tmp_path):
    result = run_wrapper("src/wrapper_eval.sh", ["/runs/x/checkpoints/last.ckpt"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "python src/eval.py ckpt_path=/runs/x/checkpoints/last.ckpt" in result.stdout


@pytest.mark.parametrize("path", WRAPPERS)
def test_a_wrapper_without_arguments_refuses(path, tmp_path):
    result = run_wrapper(path, [], tmp_path)
    assert result.returncode != 0
    assert "APPTAINER" not in result.stdout


@pytest.mark.parametrize("path,required", [("src/train_full_pipeline.sub", "EXPERIMENT"),
                                           ("src/eval.sub", "CKPT")])
def test_a_submission_without_its_target_is_refused(path, required):
    text = (REPO / path).read_text()
    assert f"if ! defined {required}" in text
    assert f'arguments  = "$({required}) $(ARGS:)"' in text


@pytest.mark.parametrize("path", SUB_FILES)
def test_the_job_is_told_where_the_repository_is(path):
    """HTCondor runs a copy of the wrapper in the node's scratch directory, where the
    repository derived from the script's own location is not the repository: the job
    looked for src/train.py in /pool/condor."""
    assert "PROJECT_DIR=$ENV(PWD)" in (REPO / path).read_text()


@pytest.mark.parametrize("path", WRAPPERS)
def test_a_wrapper_in_the_wrong_place_fails_instead_of_succeeding(path, tmp_path):
    import shutil
    import subprocess

    copy = tmp_path / "scratch" / "wrapper.sh"
    copy.parent.mkdir()
    shutil.copy(REPO / path, copy)
    result = subprocess.run(["bash", str(copy), "x"], capture_output=True, text=True,
                            env={"PATH": "/usr/bin:/bin"}, cwd=copy.parent)
    assert result.returncode != 0
    assert "submit from the repository root" in result.stderr


# ---------------------------------------------------------------------------
# Linear probes, through the evaluation wrapper
# ---------------------------------------------------------------------------


def test_the_probe_job_runs_the_probe_on_the_experiment_it_names():
    text = (REPO / "src/probe.sub").read_text()
    assert "EVAL_SCRIPT=src/eval_probes.py" in text
    assert 'arguments  = "$(CKPT) experiment=$(EXPERIMENT) $(ARGS:)"' in text
    assert "if ! defined EXPERIMENT" in text and "if ! defined CKPT" in text


def test_the_wrapper_runs_the_probe_when_asked(tmp_path, monkeypatch):
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "apptainer").write_text('#!/bin/bash\necho "APPTAINER $*"\n')
    (bin_dir / "apptainer").chmod(0o755)
    image = tmp_path / "fm.sif"
    image.write_text("")
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "FM_TESTING_IMAGE": str(image),
           "EVAL_SCRIPT": "src/eval_probes.py"}
    result = subprocess.run(["bash", str(REPO / "src/wrapper_eval.sh"), "/r/last.ckpt",
                             "experiment=fm_testing_binary"],
                            env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert ("python src/eval_probes.py ckpt_path=/r/last.ckpt experiment=fm_testing_binary"
            in result.stdout)


def test_the_wrapper_runs_nothing_else(tmp_path):
    import subprocess

    env = {"PATH": "/usr/bin:/bin", "EVAL_SCRIPT": "src/train.py"}
    result = subprocess.run(["bash", str(REPO / "src/wrapper_eval.sh"), "/r/last.ckpt"],
                            env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "EVAL_SCRIPT must be" in result.stderr


# ---------------------------------------------------------------------------
# A missing image is found before the queue, or at least before the container
#
# A probe job submitted from a terminal without FM_TESTING_IMAGE waited for a GPU and
# then failed in apptainer, on an image path inside the repository.
# ---------------------------------------------------------------------------

ALL_WRAPPERS = ("src/wrapper_full_pipeline.sh", "src/wrapper_eval.sh",
                "src/data/wrapper_vectorize.sh", "src/preprocessing/wrapper_preprocess.sh")


@pytest.mark.parametrize("path", ALL_WRAPPERS)
def test_a_wrapper_without_its_image_says_how_to_fix_it(path, tmp_path):
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "apptainer").write_text('#!/bin/bash\necho "APPTAINER $*"\n')
    (bin_dir / "apptainer").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "FM_TESTING_IMAGE": str(tmp_path / "none.sif"),
           "PROJECT_DIR": str(REPO)}
    result = subprocess.run(["bash", str(REPO / path), "a", str(REPO), "b"], env=env,
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "APPTAINER" not in result.stdout
    assert "set FM_TESTING_IMAGE" in result.stderr


@pytest.mark.parametrize("path", SUBMITTERS)
def test_nothing_is_submitted_without_an_image(path, monkeypatch, tmp_path):
    module = load(path)
    monkeypatch.setenv("FM_TESTING_IMAGE", str(tmp_path / "none.sif"))
    with pytest.raises(SystemExit, match="No container image"):
        module.require_image()
    (tmp_path / "fm.sif").write_text("")
    monkeypatch.setenv("FM_TESTING_IMAGE", str(tmp_path / "fm.sif"))
    module.require_image()
