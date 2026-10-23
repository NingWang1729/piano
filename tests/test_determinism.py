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

import os
import random
import resource
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

import piano
from piano import Composer

N_CELLS = 200
N_GENES = 50
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
MEMORY_MODES = ['CPU', 'SparseCPU', 'backed', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)]


# ---------------------------------------------------------------------------
# Helpers: small count matrix with a batch covariate, and a fixture that restores torch's process-wide settings
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(tmp_path, deterministic, memory_mode='CPU', random_seed=0, adata=None):
    return Composer(
        make_adata() if adata is None else adata, memory_mode=memory_mode, compile_model=False,
        categorical_covariate_keys=['batch'], n_top_genes=-1, max_epochs=1, batch_size=64, early_stopping=False,
        deterministic=deterministic, random_seed=random_seed, outdir=str(tmp_path),
    )


@pytest.fixture
def restore_determinism():
    # These tests turn deterministic algorithms on, which is process-wide; restore the settings for later tests
    algorithms, cudnn = torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic
    yield
    torch.use_deterministic_algorithms(algorithms)
    torch.backends.cudnn.deterministic = cudnn


def test_deterministic_false_turns_off_deterministic_algorithms(restore_determinism, tmp_path):
    make_composer(tmp_path, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic

    # Previously a later Composer with deterministic=False kept the settings an earlier Composer turned on
    make_composer(tmp_path, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_gpu_training_after_a_deterministic_composer(restore_determinism, tmp_path):
    make_composer(tmp_path, deterministic=True)

    # Previously failed in the first GPU matrix multiplication with the CuBLAS determinism error
    # ("... set CUBLAS_WORKSPACE_CONFIG ..."), as deterministic algorithms were still on
    pianist = make_composer(tmp_path, deterministic=False, memory_mode='GPU')
    pianist.run_pipeline()
    assert pianist.trained_model


def test_deterministic_true_sets_cublas_workspace_config(restore_determinism, tmp_path, monkeypatch):
    # monkeypatch restores the environment after the test (conftest.py sets the variable for the session)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    make_composer(tmp_path, deterministic=True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'


def test_deterministic_true_keeps_the_users_cublas_workspace_config(restore_determinism, tmp_path, monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    make_composer(tmp_path, deterministic=True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':16:8'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_deterministic_gpu_training_in_a_fresh_process(tmp_path):
    # torch reads CUBLAS_WORKSPACE_CONFIG once, at the first GPU matrix product in the process, and this test session
    # sets it at startup (conftest.py), so train in a separate process without it, as a user's script would start.
    # Previously failed in the first GPU matrix product with the CuBLAS determinism error
    path = tmp_path / 'adata.h5ad'
    make_adata().write_h5ad(path)
    script = (
        f"import anndata, piano; "
        f"pianist = piano.Composer(anndata.read_h5ad({str(path)!r}), memory_mode='GPU', compile_model=False, "
        f"categorical_covariate_keys=['batch'], n_top_genes=-1, max_epochs=1, batch_size=64, early_stopping=False, "
        f"deterministic=True, outdir={str(tmp_path)!r}); "
        f"pianist.run_pipeline(); print('trained:', pianist.trained_model)"
    )
    env = {key: value for key, value in os.environ.items() if key != 'CUBLAS_WORKSPACE_CONFIG'}
    env['PYTHONPATH'] = str(Path(piano.__file__).parents[1])
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env,
                            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'trained: True' in result.stdout


# ---------------------------------------------------------------------------
# random_seed=None: the Composer seeds nothing, so results follow the caller's random state
# ---------------------------------------------------------------------------
def seed_global_rngs(seed):
    # Leaves the global random number generators seeded after the test; later tests that need a seed set their own
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
@pytest.mark.parametrize('deterministic', [True, False])
def test_composer_accepts_random_seed_none(restore_determinism, tmp_path, memory_mode, deterministic):
    adata = make_adata()
    if memory_mode == 'backed':
        # Backed mode reads the counts from an h5ad file, written to tmp_path
        adata.write_h5ad(tmp_path / 'adata.h5ad')
        adata = ad.read_h5ad(tmp_path / 'adata.h5ad', backed='r')

    # Previously raised AttributeError: 'Composer' object has no attribute 'random_seed'
    pianist = make_composer(tmp_path, deterministic=deterministic, memory_mode=memory_mode, random_seed=None, adata=adata)
    assert pianist.random_seed is None


def test_composer_without_random_seed_leaves_global_rngs_unchanged(restore_determinism, tmp_path):
    seed_global_rngs(123)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

    make_composer(tmp_path, deterministic=False, random_seed=None)
    assert random.getstate() == states[0]
    np.testing.assert_equal(np.random.get_state(), states[1])
    assert torch.equal(torch.get_rng_state(), states[2])
    if torch.cuda.is_available():
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_states))


def test_training_without_random_seed_follows_the_callers_rng(restore_determinism, tmp_path):
    # One CPU epoch each: the same caller seed gives the same weights, another caller seed different ones, as neither the
    # Composer's creation nor train() reseeds the global random number generators
    def trained_weights(caller_seed):
        seed_global_rngs(caller_seed)
        pianist = make_composer(tmp_path, deterministic=False, random_seed=None)
        pianist.run_pipeline()
        return [weight.detach().cpu().clone() for weight in pianist.model.state_dict().values()]

    first, repeated, other = trained_weights(1), trained_weights(1), trained_weights(2)
    assert all(torch.equal(a, b) for a, b in zip(first, repeated))
    assert not all(torch.equal(a, b) for a, b in zip(first, other))
