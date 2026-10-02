#!/bin/bash
# =====================================================
#  HTCondor wrapper for vectorization jobs
# =====================================================

set -euo pipefail

MANIFEST_PATH="$1"
PROJECT_DIR="$2"
EXPERIMENT="${3:-}"

# The .sif is about 9 GB, which does not fit in a 10 GB AFS home, so it often lives
# elsewhere: set FM_TESTING_IMAGE to say where.
IMAGE="${FM_TESTING_IMAGE:-${PROJECT_DIR}/fm_testing.sif}"

echo "[`date`] Starting vectorization job on $(hostname)"
echo "[`date`] Running as $(whoami)"
echo "Manifest path: ${MANIFEST_PATH}"

cd ${PROJECT_DIR}

apptainer exec --cleanenv \
    --bind /eos:/eos \
    --bind /afs:/afs \
    --writable-tmpfs \
    ${IMAGE} bash -lc "python src/data/vectorize_job.py --manifest-path ${MANIFEST_PATH} ${EXPERIMENT:+experiment=$EXPERIMENT}"

EXIT_CODE=$?
echo "[`date`] Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
