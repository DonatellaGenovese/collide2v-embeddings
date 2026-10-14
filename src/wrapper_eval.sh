#!/bin/bash
# =====================================================
#  HTCondor wrapper for evaluating one checkpoint, see src/eval.sub and src/probe.sub
# =====================================================
#
# Usage: wrapper_eval.sh <checkpoint> [hydra overrides...]
#
# EVAL_SCRIPT says what to run on it: src/eval.py (the default), the test pass of
# the trained model, or src/eval_probes.py, a linear probe on its frozen embedding.
#
# Nothing here runs Python outside the container: the batch nodes have no hydra.

set -euo pipefail

CKPT="${1:?usage: wrapper_eval.sh <checkpoint> [hydra overrides...]}"
shift

echo "[`date`] Starting evaluation job on $(hostname)"
echo "[`date`] Running as $(whoami)"
SCRIPT="${EVAL_SCRIPT:-src/eval.py}"
case "${SCRIPT}" in
    src/eval.py|src/eval_probes.py) ;;
    *) echo "[wrapper] EVAL_SCRIPT must be src/eval.py or src/eval_probes.py, not ${SCRIPT}" >&2
       exit 1 ;;
esac

echo "Script: ${SCRIPT}, checkpoint: ${CKPT}, overrides: $*"

# --- Paths ---
# The submit file passes PROJECT_DIR, since HTCondor runs a copy of this script in
# the node's scratch directory; run by hand, it is wherever this script is. The
# image is about 7 GB and often lives outside the repository, hence FM_TESTING_IMAGE.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${FM_TESTING_IMAGE:-${PROJECT_DIR}/fm_testing.sif}"

if [ ! -f "${PROJECT_DIR}/${SCRIPT}" ]; then
    echo "[wrapper] no ${SCRIPT} in ${PROJECT_DIR}: submit from the repository root" >&2
    exit 1
fi
cd "${PROJECT_DIR}"

if [ ! -f "${IMAGE}" ]; then
    echo "[wrapper] no container image at ${IMAGE}: set FM_TESTING_IMAGE in the shell you" \
         "submit from, as in section 2 of the README, and submit again" >&2
    exit 1
fi

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
    "${IMAGE}" python "${SCRIPT}" ckpt_path="${CKPT}" "$@" || EXIT_CODE=$?

echo "[`date`] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
