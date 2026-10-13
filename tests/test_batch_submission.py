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
SUB_FILES = ("src/train_full_pipeline.sub", "src/eval.sub")


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
