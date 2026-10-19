"""
PIANO: Probabilistic Inference Autoencoder Networks for multi-Omics
Copyright (C) 2025 Ning Wang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

# Repository root, so the subprocesses import this checkout of piano
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Slow optional dependencies that must only load when a function needing them is used
HEAVY_MODULES = ['umap', 'pynndescent', 'faiss', 'scib_metrics', 'jax']

# Names that piano and piano.utils load lazily, with the submodule that defines each
LAZY_NAMES = {
    'faiss_knn': 'piano.utils.umap',
    'faiss_knn_to_umap_graph': 'piano.utils.umap',
    'pca_init': 'piano.utils.umap',
    'optimize_embedding': 'piano.utils.umap',
    'faiss_umap': 'piano.utils.umap',
    'plot_umaps': 'piano.utils.umap',
    'FAISS_KNN': 'piano.utils.knn',
}


def run_python(code):
    """Run code in a fresh interpreter (so sys.modules starts empty) and return its stdout."""
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(code)],
        cwd=REPO_ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def loaded_heavy_modules(import_statement):
    """Return the heavy modules present in sys.modules after running import_statement."""
    stdout = run_python(f'''
        import json
        import sys
        {import_statement}
        print(json.dumps([m for m in {HEAVY_MODULES!r} if m in sys.modules]))
    ''')
    return json.loads(stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# Importing piano does not load the slow optional dependencies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('import_statement', ['import piano', 'from piano import Composer', 'import piano.utils'])
def test_import_piano_skips_heavy_modules(import_statement):
    assert loaded_heavy_modules(import_statement) == []


# ---------------------------------------------------------------------------
# The lazily loaded names still resolve, from piano and piano.utils
# ---------------------------------------------------------------------------
def test_lazy_names_resolve():
    stdout = run_python(f'''
        import importlib
        import piano
        import piano.utils
        from piano import *
        for name, module in {LAZY_NAMES!r}.items():
            expected = getattr(importlib.import_module(module), name)
            assert getattr(piano, name) is expected, name
            assert getattr(piano.utils, name) is expected, name
            assert globals()[name] is expected, name
        from piano import plot_umaps, FAISS_KNN
        from piano.utils import faiss_umap
        print('ok')
    ''')
    assert stdout.strip().splitlines()[-1] == 'ok'


def test_unknown_attribute_raises():
    stdout = run_python('''
        import piano
        try:
            piano.not_a_piano_function
        except AttributeError as error:
            print(error)
    ''')
    assert "module 'piano' has no attribute 'not_a_piano_function'" in stdout


# ---------------------------------------------------------------------------
# The command line interface loads scib_metrics (and JAX) only when benchmarking
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('import_statement', [
    'import piano.__main__',
    # The example script, run under a name other than '__main__' so its pipeline does not start
    "import runpy; runpy.run_path('run_piano_integration.py')",
])
def test_cli_import_skips_scib_metrics(import_statement):
    # Importing the module defines main() and its imports without running the pipeline
    loaded = loaded_heavy_modules(import_statement)
    assert 'scib_metrics' not in loaded
    assert 'jax' not in loaded


@pytest.mark.parametrize('load_main', [
    'from piano.__main__ import main',
    "import runpy; main = runpy.run_path('run_piano_integration.py')['main']",
])
def test_cli_without_scib_metrics_fails_at_startup(load_main):
    # With --scib_benchmarking, a missing scib_metrics must fail before anything else runs (previously it failed
    # after training). The arguments hold only scib_benchmarking, so reading any other argument first would raise
    # an AttributeError instead
    stdout = run_python(f'''
        import argparse
        import sys
        sys.modules['scib_metrics'] = None  # Makes `import scib_metrics` raise ImportError, as if it were not installed
        {load_main}
        try:
            main(argparse.Namespace(scib_benchmarking=True))
        except Exception as error:
            print(type(error).__name__)
    ''')
    assert stdout.strip().splitlines()[-1] in ('ImportError', 'ModuleNotFoundError')
