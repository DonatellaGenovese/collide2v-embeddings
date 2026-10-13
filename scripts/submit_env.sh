# Source this on lxplus, outside the container, before running the submit scripts:
#
#     source scripts/submit_env.sh
#     python scripts/submit_vectorization_jobs.py experiment=<name>
#
# The submit scripts need two things that never exist in the same place: the project's
# own code, to plan the jobs, which imports awkward, numpy and torch; and condor_submit,
# which exists on lxplus but not inside the container. So they run on lxplus itself, with
# the heavy libraries taken from an LCG view on CVMFS, which installs nothing, and a small
# environment on top for the few packages LCG does not have. The jobs themselves still run
# inside the container.
#
# The environment is created on first use, about 20 MB, in COLLIDE_SUBMIT_ENV.

LCG_VIEW="${COLLIDE_LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc16-opt}"
SUBMIT_ENV="${COLLIDE_SUBMIT_ENV:-$HOME/.venvs/collide2v-submit}"

if [ ! -f "$LCG_VIEW/setup.sh" ]; then
    echo "submit_env.sh: no LCG view at $LCG_VIEW. Is this lxplus, with CVMFS mounted?" >&2
    return 1
fi
source "$LCG_VIEW/setup.sh"

if [ ! -x "$SUBMIT_ENV/bin/python" ]; then
    echo "submit_env.sh: creating the submission environment in $SUBMIT_ENV"
    # Versions as in requirements.txt; antlr4 4.9.3 is the one omegaconf is built
    # against, and LCG ships a newer one it cannot read.
    python3 -m venv --system-site-packages "$SUBMIT_ENV" \
        && "$SUBMIT_ENV/bin/pip" install --quiet --no-user \
            hydra-core==1.3.2 hydra-colorlog==1.2.0 rootutils antlr4-python3-runtime==4.9.3 \
        || { echo "submit_env.sh: could not create $SUBMIT_ENV" >&2; rm -rf "$SUBMIT_ENV"; return 1; }
fi
source "$SUBMIT_ENV/bin/activate"

# The LCG view puts its packages on PYTHONPATH, which Python searches before the
# environment's own, so its antlr4 would still win. Put ours first.
PYTHONPATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'):$PYTHONPATH"
export PYTHONPATH

echo "submit_env.sh: ready. Run the submit scripts from the repository root."
