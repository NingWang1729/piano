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


import ast
import importlib.metadata
import os
import re
import subprocess
import sys
import textwrap
import tomllib

from setuptools import find_packages

# Repository root, so the tests read this checkout's pyproject.toml and source files
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_pyproject():
    with open(os.path.join(REPO_ROOT, 'pyproject.toml'), 'rb') as f:
        return tomllib.load(f)


def distribution_name(requirement):
    """Normalized distribution name of a requirement string, e.g. 'torch>=2.2,<2.8' -> 'torch' (PEP 503)."""
    name = re.match(r'[A-Za-z0-9._-]+', requirement).group(0)
    return re.sub(r'[-_.]+', '-', name).lower()


# Modules whose imports belong to an optional extra: they load only when their helpers are first used (see
# test_lazy_imports.py), so a plain install does not need faiss. piano/utils/umap.py imports faiss only inside its
# FAISS helpers, as plot_umaps and the command line use that module too
OPTIONAL_MODULES = {
    os.path.join('piano', 'utils', 'knn.py'): 'misc extra (faiss-cpu)',
    # Loaded only by the SparseGPU memory mode; triton comes with torch's CUDA builds on Linux and exists nowhere else
    os.path.join('piano', 'utils', 'triton_sparse.py'): 'triton (with torch on Linux)',
}


def piano_source_files():
    """Every Python file of the piano package, as paths relative to the repository root, except OPTIONAL_MODULES."""
    paths = []
    for root, _, files in os.walk(os.path.join(REPO_ROOT, 'piano')):
        paths += [os.path.relpath(os.path.join(root, name), REPO_ROOT) for name in files if name.endswith('.py')]
    return sorted(path for path in paths if path not in OPTIONAL_MODULES)


def catches_import_errors(handler):
    """Whether an except clause catches a failed import: bare except, ImportError, ModuleNotFoundError or Exception."""
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    caught = ('ImportError', 'ModuleNotFoundError', 'Exception')
    return any(name is None or (isinstance(name, ast.Name) and name.id in caught) for name in names)


def module_level_imports(path):
    """Top-level names of the modules a file needs at import time: absolute imports outside functions and classes,
    except those in a try block that catches a failed import (optional, e.g. rapids_singlecell in piano/__main__.py)."""
    with open(os.path.join(REPO_ROOT, path)) as f:
        tree = ast.parse(f.read())
    names = set()
    # Statements at module level, including those nested in module-level if/try blocks, but not in def or class
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split('.')[0])
        elif isinstance(node, ast.If):
            pending += node.body + node.orelse
        elif isinstance(node, ast.Try):
            # The try body is optional when a handler catches a failed import; the rest always runs or may run
            if not any(catches_import_errors(handler) for handler in node.handlers):
                pending += node.body
            pending += node.orelse + node.finalbody
            pending += [statement for handler in node.handlers for statement in handler.body]
    return names


# ---------------------------------------------------------------------------
# Guard: every third-party package piano's modules import is a declared dependency, not one that only arrives because
# another dependency happens to need it (fails before this change: numba, h5py, statsmodels, tqdm)
# ---------------------------------------------------------------------------
def test_imported_packages_are_declared_dependencies():
    declared = {distribution_name(requirement) for requirement in read_pyproject()['project']['dependencies']}
    distributions_of_module = importlib.metadata.packages_distributions()

    undeclared = {}
    for path in piano_source_files():
        for module in module_level_imports(path) - set(sys.stdlib_module_names) - {'piano'}:
            # The installed distribution(s) that provide the module, e.g. skmisc -> scikit-misc
            distributions = {distribution_name(d) for d in distributions_of_module.get(module, [module])}
            if not distributions & declared:
                undeclared.setdefault(module, set()).add(path)
    assert not undeclared, (
        'Imported by piano but not in pyproject.toml [project] dependencies: '
        + '; '.join(f'{module} (in {", ".join(sorted(files))})' for module, files in sorted(undeclared.items()))
    )


def test_piano_does_not_depend_on_pyro():
    """pyro-ppl is no runtime dependency: piano's ZINB is plain torch (SCRUM-46); tests use pyro as the reference."""
    project = read_pyproject()['project']
    assert 'pyro-ppl' not in {distribution_name(requirement) for requirement in project['dependencies']}
    for path in piano_source_files():
        assert 'pyro' not in module_level_imports(path), path


# ---------------------------------------------------------------------------
# Packaging is configured in pyproject.toml alone: its package discovery finds every piano package and nothing else
# ---------------------------------------------------------------------------
def test_pyproject_alone_configures_the_build():
    pyproject = read_pyproject()
    assert pyproject['build-system']['build-backend'] == 'setuptools.build_meta'
    assert not os.path.exists(os.path.join(REPO_ROOT, 'setup.py'))

    find = pyproject['tool']['setuptools']['packages']['find']
    packages = find_packages(REPO_ROOT, include=find['include'])
    # Every folder under piano/ with an __init__.py, as dotted names
    expected = sorted(
        os.path.relpath(root, REPO_ROOT).replace(os.sep, '.')
        for root, _, files in os.walk(os.path.join(REPO_ROOT, 'piano'))
        if '__init__.py' in files
    )
    assert sorted(packages) == expected


# ---------------------------------------------------------------------------
# A plain install has no faiss (it is in the misc extra): the command line and plot_umaps still work, and the FAISS
# helpers say what to install (fails before this change: piano/utils/umap.py imported faiss at the top)
# ---------------------------------------------------------------------------
def run_without(module, code):
    """Run code in a fresh interpreter in which `import <module>` fails, as if it were not installed; return the result."""
    preamble = f'import sys\nsys.modules["{module}"] = None  # Makes `import {module}` raise ImportError\n'
    return subprocess.run(
        [sys.executable, '-c', preamble + textwrap.dedent(code)],
        cwd=REPO_ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True, timeout=600,
    )


def test_command_line_and_plot_umaps_work_without_faiss():
    result = run_without('faiss', '''
        import runpy
        from piano import plot_umaps
        sys.argv = ['piano', '--help']
        try:
            runpy.run_module('piano', run_name='__main__')
        except SystemExit as exit:
            assert exit.code == 0, exit.code
    ''')
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_faiss_helpers_name_the_extra_without_faiss():
    result = run_without('faiss', '''
        import numpy as np
        from piano import faiss_knn
        try:
            faiss_knn(np.zeros((10, 2), dtype=np.float32), k=2)
        except ImportError as error:
            print(error)
    ''')
    assert result.returncode == 0, result.stderr
    assert 'piano-integration[misc]' in result.stdout


# ---------------------------------------------------------------------------
# Without triton (anywhere but Linux), piano imports and trains in the other memory modes, and SparseGPU says what
# it needs (fails before this change: `import piano` imported piano.utils.triton_sparse, which imports triton)
# ---------------------------------------------------------------------------
def test_piano_trains_without_triton(tmp_path):
    # Training writes its checkpoints to pytest's tmp_path (a fresh directory per test)
    result = run_without('triton', '''
        import anndata as ad
        import numpy as np
        import pandas as pd
        from piano import Composer
        rng = np.random.default_rng(0)
        adata = ad.AnnData(
            X=rng.poisson(1.0, size=(64, 20)).astype(np.float32),
            obs=pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], 64))}, index=[str(i) for i in range(64)]),
        )
        adata.var_names = [f'gene_{i}' for i in range(20)]
        for memory_mode in ['CPU', 'SparseCPU']:
            pianist = Composer(adata, memory_mode=memory_mode, categorical_covariate_keys=['batch'], n_top_genes=-1,
                               max_epochs=1, batch_size=32, compile_model=False, deterministic=False,
                               outdir=f'OUTDIR/{memory_mode}')
            pianist.run_pipeline()
            assert pianist.get_latent_representation().shape[0] == 64
        print('ok')
    '''.replace('OUTDIR', str(tmp_path)))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == 'ok'


def test_sparse_gpu_names_triton_without_triton():
    result = run_without('triton', '''
        from piano.utils.data import _import_sparse_triton_matrix
        try:
            _import_sparse_triton_matrix()
        except ImportError as error:
            print(error)
    ''')
    assert result.returncode == 0, result.stderr
    assert 'SparseGPU memory mode needs triton' in result.stdout


# ---------------------------------------------------------------------------
# pandas 3 is not supported yet (SCRUM-194), so piano itself excludes it: only the newest anndata caps pandas below
# 3, and an install with pandas 3 already present resolves to an older anndata without that cap
# ---------------------------------------------------------------------------
def test_pandas_3_is_excluded():
    from packaging.requirements import Requirement
    from packaging.version import Version

    requirements = [Requirement(r) for r in read_pyproject()['project']['dependencies']]
    pandas = next(r for r in requirements if r.name == 'pandas')
    # An upper bound below 3 (e.g. <3), so no pandas 3 release can be installed
    assert any(spec.operator == '<' and Version(spec.version) <= Version('3') for spec in pandas.specifier), pandas
