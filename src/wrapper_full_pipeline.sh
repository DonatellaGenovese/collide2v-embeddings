#!/bin/bash
# =====================================================
#  HTCondor wrapper for one training run, see src/train_full_pipeline.sub
# =====================================================
#
# Usage: wrapper_full_pipeline.sh <experiment> [hydra overrides...]
#
# Nothing here runs Python outside the container: the batch nodes have no hydra, and
# a script that needed it failed before training started.

set -euo pipefail

EXPERIMENT="${1:?usage: wrapper_full_pipeline.sh <experiment> [hydra overrides...]}"
shift

echo "[`date`] Starting training job on $(hostname)"
echo "[`date`] Running as $(whoami)"
echo "Experiment: ${EXPERIMENT}, overrides: $*"

# --- Paths ---
# The submit file passes PROJECT_DIR, since HTCondor runs a copy of this script in
# the node's scratch directory; run by hand, it is wherever this script is. The
# image is about 7 GB and often lives outside the repository, hence FM_TESTING_IMAGE.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${FM_TESTING_IMAGE:-${PROJECT_DIR}/fm_testing.sif}"

if [ ! -f "${PROJECT_DIR}/src/train.py" ]; then
    echo "[wrapper] no src/train.py in ${PROJECT_DIR}: submit from the repository root" >&2
    exit 1
fi
cd "${PROJECT_DIR}"

# --- GPU ---
# Decided by what the node has, which is what the submit file asked for, rather than
# by composing the config, which would need hydra outside the container.
NV_FLAG=""
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    echo "[wrapper] GPU found → running with --nv"
    NV_FLAG="--nv"
else
    echo "[wrapper] no GPU on this node → running without --nv"
fi

# `|| EXIT_CODE=$?`: under set -e a failure would end the script here, before the
# line below says how the job ended.
EXIT_CODE=0
apptainer exec ${NV_FLAG} \
    --bind /afs:/afs \
    --bind /eos:/eos \
    --writable-tmpfs \
    "${IMAGE}" python src/train.py experiment="${EXPERIMENT}" "$@" || EXIT_CODE=$?

echo "[`date`] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
