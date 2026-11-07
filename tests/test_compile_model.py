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

import contextlib
import json
import os
import re
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import torch._dynamo
import torch._inductor.config
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50
# 200 cells in batches of 20: every training batch is full in PIANO mode (200 cells) and in scVI mode (its 90% split,
# 180 cells), so each mode compiles one graph only, which keeps these tests fast
BATCH_SIZE = 20
# Repository root, so the subprocesses import this checkout of piano and find run_piano_integration.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CLI_ENTRY_POINTS = {
    'piano': [sys.executable, '-m', 'piano'],
    'run_piano_integration': [sys.executable, os.path.join(REPO_ROOT, 'run_piano_integration.py')],
}


# ---------------------------------------------------------------------------
# Helpers: small count matrix with a batch covariate, and a small Composer on the CPU
# ---------------------------------------------------------------------------
def make_adata():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(tmp_path, mode='piano', **kwargs):
    # deterministic is left at its default (True), the setting CPU users get
    mode_kwargs = {'mode': 'scvi', 'batch_key': 'batch'} if mode == 'scvi' else {'categorical_covariate_keys': ['batch']}
    return Composer(make_adata(), n_top_genes=-1, n_hidden=16, n_layers=1, latent_size=4, max_epochs=2,
                    batch_size=BATCH_SIZE, outdir=str(tmp_path / 'outdir'), **mode_kwargs, **kwargs)


@pytest.fixture
def no_gpu(monkeypatch):
    """Pretend that no CUDA GPU is available, so the Composer trains on the CPU also on a GPU node."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


@pytest.fixture
def fresh_compiler_state():
    """Start from no compiled graphs, and count what dynamo compiles."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    yield
    torch._dynamo.reset()


@pytest.fixture
def restore_determinism():
    # deterministic=True (the default) sets deterministic algorithms and cuDNN flags, which are process-wide; restore
    # them for later tests
    algorithms = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic, cudnn_benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(algorithms)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_deterministic, cudnn_benchmark


@pytest.fixture
def no_cpp_compiler():
    """Make torch find no C++ compiler: inductor searches only torch._inductor.config.cpp.cxx."""
    # torch's own config patch, not monkeypatch: torch's config proxy resolves cxx dynamically, so since pytest 9.2.0
    # monkeypatch's undo deletes the setting instead of restoring it, and every later Composer fails (SCRUM-209)
    with torch._inductor.config.patch({'cpp.cxx': ('no_such_cpp_compiler',)}):
        yield


def n_compiled_graphs():
    # Dynamo's count of the graphs it captured and compiled (a torch internal, but the only direct evidence that
    # torch.compile ran rather than the eager train step)
    return torch._dynamo.utils.counters['stats']['unique_graphs']


# ---------------------------------------------------------------------------
# compile_model=True compiles the training step on the CPU too
# ---------------------------------------------------------------------------
# Trains a compiled Composer on the CPU in a fresh interpreter with CUDA hidden, and prints what the test checks as
# JSON on the last line. Not in the pytest process with torch.cuda.is_available patched: inductor caches for the whole
# process whether a GPU can use triton (torch.utils._triton.has_triton, an lru_cache), so compiling there would make
# later GPU compilation in the same session fail with TritonMissing
TRAIN_COMPILED_ON_CPU = """
import json, pathlib, sys, warnings
import numpy as np
sys.path.insert(0, {tests_dir!r})
from test_compile_model import make_composer, n_compiled_graphs
with warnings.catch_warnings(record=True) as record:
    warnings.simplefilter('always')
    pianist = make_composer(pathlib.Path({tmp_path!r}), mode={mode!r}, memory_mode={memory_mode!r}, compile_model=True)
compile_model = pianist.compile_model
pianist.run_pipeline()
latent = pianist.get_latent_representation()
print(json.dumps({{
    'warnings': [str(warning.message) for warning in record], 'compile_model': compile_model,
    'device': pianist.device, 'graphs': n_compiled_graphs(), 'latent_shape': list(latent.shape),
    'latent_finite': bool(np.isfinite(latent).all()),
}}))
"""


@pytest.mark.parametrize('mode, memory_mode', [('piano', 'CPU'), ('piano', 'GPU'), ('scvi', 'CPU')])
def test_compile_model_compiles_training_on_cpu(tmp_path, mode, memory_mode):
    # Previously the training step was compiled only with a CUDA GPU, and memory_mode='GPU' falling back to 'CPU' also
    # turned compile_model off. A fresh inductor cache in tmp_path makes the run compile for real (a cache shared with
    # other runs could hold kernels built for another thread count)
    code = TRAIN_COMPILED_ON_CPU.format(tests_dir=TESTS_DIR, tmp_path=str(tmp_path), mode=mode, memory_mode=memory_mode)
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                 'TORCHINDUCTOR_CACHE_DIR': str(tmp_path / 'inductor_cache')})
    assert result.returncode == 0, result.stderr[-3000:]
    run = json.loads(result.stdout.strip().splitlines()[-1])

    fallback_warnings = [message for message in run['warnings'] if "using memory_mode='CPU'" in message]
    assert len(fallback_warnings) == (1 if memory_mode == 'GPU' else 0)
    assert (run['compile_model'], run['device']) == (True, 'cpu')
    assert run['graphs'] > 0
    assert run['latent_shape'] == [N_CELLS, 4]
    assert run['latent_finite']


def test_compile_model_false_does_not_compile_on_cpu(no_gpu, fresh_compiler_state, restore_determinism, tmp_path):
    # Coverage: compile_model=False trains the eager step
    pianist = make_composer(tmp_path, memory_mode='CPU', compile_model=False)
    pianist.run_pipeline()
    assert n_compiled_graphs() == 0


# ---------------------------------------------------------------------------
# Compiling on the CPU needs a C++ compiler: without one, the Composer raises when it is created
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('memory_mode', ['CPU', 'SparseCPU', 'GPU'])
def test_missing_cpp_compiler_raises_at_creation(no_gpu, no_cpp_compiler, restore_determinism, tmp_path, memory_mode):
    # Previously no error at creation (nothing was compiled on the CPU)
    with pytest.warns(UserWarning, match="using memory_mode='CPU'") if memory_mode == 'GPU' else contextlib.nullcontext():
        with pytest.raises(RuntimeError, match=r'C\+\+ compiler.*compile_model=False'):
            make_composer(tmp_path, memory_mode=memory_mode, compile_model=True)


def test_missing_cpp_compiler_without_compilation_trains(no_gpu, no_cpp_compiler, fresh_compiler_state, restore_determinism,
                                                         tmp_path):
    # Coverage: compile_model=False needs no C++ compiler
    pianist = make_composer(tmp_path, memory_mode='CPU', compile_model=False)
    pianist.run_pipeline()
    assert pianist.get_latent_representation().shape == (N_CELLS, 4)


def test_load_without_cpp_compiler_keeps_compile_model_for_inference(no_gpu, no_cpp_compiler, restore_determinism, tmp_path):
    # Coverage: Composer.load does not check for a C++ compiler, since a loaded Composer has no training data to train
    # on and inference is never compiled. Trained uncompiled to keep the test fast, then saved with compile_model=True
    pianist = make_composer(tmp_path, memory_mode='CPU', compile_model=False)
    pianist.run_pipeline()
    pianist.compile_model = True
    pianist.save(tmp_path / 'pianist.pkl')

    loaded = Composer.load(tmp_path / 'pianist.pkl')
    assert loaded.compile_model is True
    assert loaded.get_latent_representation(make_adata()).shape == (N_CELLS, 4)


# A test using no_cpp_compiler, then one checking that torch's C++ compiler setting is back, run by pytest in a fresh
# interpreter so that the second test sees exactly what the fixture's teardown left
FIXTURE_RESTORES_CPP_COMPILER = """
import sys
sys.path[:0] = [{tests_dir!r}, {repo_root!r}]
import torch._inductor.config
from test_compile_model import no_cpp_compiler

ORIGINAL_CXX = torch._inductor.config.cpp.cxx


def test_hides_the_compiler(no_cpp_compiler):
    assert torch._inductor.config.cpp.cxx == ('no_such_cpp_compiler',)


def test_compiler_is_back_after_the_fixture():
    assert torch._inductor.config.cpp.cxx == ORIGINAL_CXX
"""


def test_no_cpp_compiler_fixture_restores_the_compiler_setting(tmp_path):
    # Under pytest 9.2.0, monkeypatch's undo deleted torch._inductor.config.cpp.cxx, so every later Composer in the run
    # failed with "torch._inductor.config.cpp.cxx does not exist" (SCRUM-209)
    test_file = tmp_path / 'test_fixture_teardown.py'
    test_file.write_text(FIXTURE_RESTORES_CPP_COMPILER.format(tests_dir=TESTS_DIR, repo_root=REPO_ROOT))
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', str(test_file), '-p', 'no:cacheprovider', '-q'],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert '2 passed' in result.stdout


# ---------------------------------------------------------------------------
# Both command-line scripts: --compile_model False trains without a C++ compiler on the CPU
# ---------------------------------------------------------------------------
# --compile_model False in both scripts; the default in one only, as both pass the flag the same way and each run costs
# seconds of imports
@pytest.mark.parametrize('entry_point, compile_args, expected_error', [
    ('piano', [], 'RuntimeError: compile_model=True compiles the training step on the CPU'),
    # The Composer gets past the C++ compiler check and stops at the next check, as the batch is larger than the data
    ('piano', ['--compile_model', 'False'], 'ValueError: .*batch_size'),
    ('run_piano_integration', ['--compile_model', 'False'], 'ValueError: .*batch_size'),
], ids=['piano-default', 'piano-compile_model_false', 'run_piano_integration-compile_model_false'])
def test_cli_compile_model_on_cpu_without_cpp_compiler(tmp_path, entry_point, compile_args, expected_error):
    # The default batch size (512) is larger than the 200 cells, so the run stops at Composer creation either way, before
    # any training: which error it stops on shows whether the Composer was asked to compile. CXX names no compiler, and
    # CUDA is hidden so the Composer uses the CPU
    adata_path = str(tmp_path / 'adata.h5ad')
    make_adata().write_h5ad(adata_path)
    args = ['--adata_train_list', adata_path, '--adata_valid', adata_path, '--outdir', str(tmp_path / 'outdir'),
            '--memory_mode', 'CPU', '--n_top_genes', '-1', '--categorical_covariate_keys', 'batch', '--batch_key', 'batch']
    result = subprocess.run(CLI_ENTRY_POINTS[entry_point] + args + compile_args, cwd=REPO_ROOT, capture_output=True,
                            text=True, timeout=600,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'CXX': 'no_such_cpp_compiler'})
    assert result.returncode != 0
    assert re.search(expected_error, result.stderr), result.stderr[-2000:]
