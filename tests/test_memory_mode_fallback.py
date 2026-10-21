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

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from piano import Composer
from piano.utils.data import SparseCPUAnnDataset

N_CELLS = 200
N_GENES = 50
SPARSE_CONTINUOUS_COVARIATES = [('cell_type', 'score')]


# ---------------------------------------------------------------------------
# Helpers: small CSR count matrix with a batch covariate and a sparse continuous covariate (category and value columns)
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS)),
        'cell_type': pd.Categorical(rng.choice(['x', 'y', 'z'], N_CELLS)),
        'score': rng.random(N_CELLS).astype(np.float32),
    }, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(X), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(outdir, memory_mode='CPU', adata=None, **kwargs):
    kwargs.setdefault('categorical_covariate_keys', ['batch'])
    kwargs.setdefault('max_epochs', 1)
    return Composer(
        make_adata() if adata is None else adata, memory_mode=memory_mode, compile_model=False, n_top_genes=-1,
        batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir), **kwargs,
    )


@pytest.fixture
def no_gpu(monkeypatch):
    """Pretend that no CUDA GPU is available, so the fallback is tested (and fails on old code) on GPU nodes too."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


def memory_mode_warnings(record):
    """The recorded warnings about the memory mode, out of all recorded warnings."""
    return [warning for warning in record if 'memory_mode' in str(warning.message)]


# ---------------------------------------------------------------------------
# Composer creation: GPU memory modes fall back to the matching CPU memory mode, with a warning
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('memory_mode', ['SparseGPU', 'sparsegpu'])
def test_sparse_gpu_falls_back_to_sparse_cpu(memory_mode, no_gpu, tmp_path):
    # Previously fell back to dense 'CPU', so the CSR counts were made dense, with only a printed notice.
    # A memory mode in any case is canonicalized before the fallback, so the warning uses the canonical spelling
    with pytest.warns(UserWarning, match="memory_mode='SparseGPU'.*'SparseCPU'"):
        pianist = make_composer(tmp_path, memory_mode)
    assert pianist.memory_mode == 'SparseCPU'
    assert pianist.device == 'cpu'

    # The training counts stay sparse
    pianist.initialize_features()
    pianist.prepare_data()
    assert isinstance(pianist.train_adataset, SparseCPUAnnDataset)


def test_gpu_falls_back_to_cpu_with_warning(no_gpu, tmp_path):
    # Previously fell back to 'CPU' with only a printed notice
    with pytest.warns(UserWarning, match="memory_mode='GPU'.*'CPU'"):
        pianist = make_composer(tmp_path, 'GPU')
    assert pianist.memory_mode == 'CPU'
    assert pianist.device == 'cpu'


def test_fallback_warning_points_at_caller(no_gpu, tmp_path):
    # The warning names the user's Composer(...) line (here, in make_composer in this file), not piano's code
    with pytest.warns(UserWarning, match="memory_mode='SparseGPU'") as record:
        make_composer(tmp_path, 'SparseGPU')
    assert [warning.filename for warning in memory_mode_warnings(record)] == [__file__]


def test_fallback_trains_and_embeds_training_data(no_gpu, tmp_path):
    with pytest.warns(UserWarning, match="memory_mode='SparseGPU'.*'SparseCPU'"):
        pianist = make_composer(tmp_path, 'SparseGPU')
    pianist.run_pipeline()
    assert pianist.trained_model
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model_kwargs['latent_size'])


@pytest.mark.parametrize('memory_mode', ['CPU', 'SparseCPU', 'backed'])
def test_cpu_modes_do_not_warn(memory_mode, no_gpu, tmp_path):
    adata = make_adata()
    if memory_mode == 'backed':
        adata.write_h5ad(tmp_path / 'adata.h5ad')
        adata = ad.read_h5ad(tmp_path / 'adata.h5ad', backed='r')

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        pianist = make_composer(tmp_path, memory_mode, adata=adata)
    assert memory_mode_warnings(record) == []
    assert pianist.memory_mode == memory_mode


def test_sparse_gpu_fallback_accepts_sparse_continuous_covariates(no_gpu, tmp_path):
    # Previously raised "only supported in SparseCPU memory mode", because the check ran on the dense 'CPU' fallback.
    # The check now runs on the memory mode actually used, SparseCPU, which supplies sparse continuous covariates
    with pytest.warns(UserWarning, match="memory_mode='SparseGPU'.*'SparseCPU'"):
        pianist = make_composer(tmp_path, 'SparseGPU', sparse_continuous_covariate_keys=SPARSE_CONTINUOUS_COVARIATES)
    assert pianist.memory_mode == 'SparseCPU'
    assert pianist.sparse_continuous_covariate_keys == SPARSE_CONTINUOUS_COVARIATES
