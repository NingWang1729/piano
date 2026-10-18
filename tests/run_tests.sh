#!/bin/bash
# Run the unit tests in the active Python environment (activate the piano environment first).
# GPU-only tests are skipped automatically when no GPU is available.
#
# Usage: bash tests/run_tests.sh [extra pytest arguments]
#   e.g. bash tests/run_tests.sh -k umap
set -euo pipefail

# Fail early with install instructions if the active environment has no pytest
if ! python -c "import pytest" 2>/dev/null; then
    echo "Error: pytest is not installed for $(command -v python). Install it with: pip install pytest" >&2
    exit 1
fi

# Run from the repository root; python -m puts it first on sys.path, so `import piano` uses this checkout
cd "$(dirname "$(realpath "$0")")/.."
python -m pytest tests/ "$@"
