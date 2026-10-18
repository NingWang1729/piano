#!/bin/bash
# Submit the unit tests as one CPU-only Slurm job (a single pytest worker).
# The job uses the Python environment active at submission; activate the piano environment first.
# GPU-only tests are skipped. The log is written to tests/logs/pytest_<job id>.log
#
# Usage: bash tests/submit_tests_slurm.sh [extra pytest arguments]
#   e.g. bash tests/submit_tests_slurm.sh -k stratified
set -euo pipefail

TESTS_DIR="$(dirname "$(realpath "$0")")"
mkdir -p "${TESTS_DIR}/logs"

# sbatch runs a copy of a submitted script from a spool folder, so call run_tests.sh by its real path via --wrap.
# CUDA_VISIBLE_DEVICES is emptied so the CPU code paths are tested even if the job lands on a GPU node.
sbatch \
    --job-name=piano_tests \
    --cpus-per-task=8 \
    --mem=32G \
    --time=00:30:00 \
    --output="${TESTS_DIR}/logs/pytest_%j.log" \
    --wrap="CUDA_VISIBLE_DEVICES= bash $(printf '%q ' "${TESTS_DIR}/run_tests.sh" "$@")"
