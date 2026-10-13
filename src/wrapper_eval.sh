#!/bin/bash
# =====================================================
#  HTCondor wrapper for evaluating one checkpoint, see src/eval.sub
# =====================================================
#
# Usage: wrapper_eval.sh <checkpoint> [hydra overrides...]
#
# Nothing here runs Python outside the container: the batch nodes have no hydra.

set -euo pipefail

CKPT="${1:?usage: wrapper_eval.sh <checkpoint> [hydra overrides...]}"
shift

echo "[`date`] Starting evaluation job on $(hostname)"
echo "[`date`] Running as $(whoami)"
echo "Checkpoint: ${CKPT}, overrides: $*"

# --- Paths ---
# The submit file passes PROJECT_DIR, since HTCondor runs a copy of this script in
# the node's scratch directory; run by hand, it is wherever this script is. The
# image is about 7 GB and often lives outside the repository, hence FM_TESTING_IMAGE.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${FM_TESTING_IMAGE:-${PROJECT_DIR}/fm_testing.sif}"

if [ ! -f "${PROJECT_DIR}/src/eval.py" ]; then
    echo "[wrapper] no src/eval.py in ${PROJECT_DIR}: submit from the repository root" >&2
    exit 1
fi
cd "${PROJECT_DIR}"

# --- GPU ---
# Decided by what the node has, which is what the submit file asked for.
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
    "${IMAGE}" python src/eval.py ckpt_path="${CKPT}" "$@" || EXIT_CODE=$?

echo "[`date`] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
