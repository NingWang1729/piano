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


import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50
SMALL_MODEL = {'n_hidden': 16, 'n_layers': 1, 'latent_size': 8}  # Fast to train; the zero-cell checks are what is tested
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
MEMORY_MODES = ['CPU', 'SparseCPU', 'backed', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)]


# ---------------------------------------------------------------------------
# Helpers: a small count matrix with a batch covariate, or the same with zero cells, in memory or read backed
# ---------------------------------------------------------------------------
def make_adata(n_cells=N_CELLS, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(n_cells, N_GENES)).astype(np.float32)
    # The categories are fixed, so an AnnData with zero cells keeps them, as a subset of a real dataset would
    batch = pd.Categorical(rng.choice(['a', 'b'], n_cells), categories=['a', 'b'])
    obs = pd.DataFrame({'batch': batch}, index=[str(i) for i in range(n_cells)])
    adata = ad.AnnData(X=csr_matrix(X), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def read_backed(adata, path):
    """Write adata to path and read it back backed, so its counts stay on disk."""
    adata.write_h5ad(path)
    return ad.read_h5ad(path, backed='r')


def make_training_adatas(memory_mode, tmp_path, n_cells_list):
    """One AnnData per entry of n_cells_list, read backed in backed mode."""
    adatas = [make_adata(n_cells, seed=index) for index, n_cells in enumerate(n_cells_list)]
    if memory_mode == 'backed':
        adatas = [read_backed(adata, tmp_path / f'train_{index}.h5ad') for index, adata in enumerate(adatas)]
    return adatas


def make_composer(adata, memory_mode, outdir, categorical_covariate_keys=None):
    return Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=categorical_covariate_keys,
        n_top_genes=-1, max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir),
        **SMALL_MODEL,
    )


# ---------------------------------------------------------------------------
# Training: a zero-cell AnnData raises when the Composer is created, with or without covariates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('covariates', [None, ['batch']], ids=['no_covariates', 'batch'])
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_zero_cell_training_adata_raises_at_creation(memory_mode, covariates, tmp_path):
    adatas = make_training_adatas(memory_mode, tmp_path, [N_CELLS, 0])
    with pytest.raises(ValueError, match=r'adata 1 has no cells \(n_obs is 0\)'):
        make_composer(adatas, memory_mode, tmp_path, categorical_covariate_keys=covariates)


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_single_zero_cell_training_adata_raises_at_creation(memory_mode, tmp_path):
    adatas = make_training_adatas(memory_mode, tmp_path, [0])
    with pytest.raises(ValueError, match=r'adata 0 has no cells \(n_obs is 0\)'):
        make_composer(adatas[0], memory_mode, tmp_path, categorical_covariate_keys=['batch'])


# ---------------------------------------------------------------------------
# Query: a zero-cell AnnData raises in get_latent_representation and get_counterfactual, before any dataset is built
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def pianist(tmp_path_factory):
    """A model trained for one CPU epoch."""
    composer = make_composer(make_adata(), 'CPU', tmp_path_factory.mktemp('empty_cpu'), categorical_covariate_keys=['batch'])
    composer.run_pipeline()
    return composer


@pytest.fixture(scope='module')
def backed_pianist(tmp_path_factory):
    """A model trained for one epoch in backed mode."""
    outdir = tmp_path_factory.mktemp('empty_backed')
    composer = make_composer(read_backed(make_adata(), outdir / 'train.h5ad'), 'backed', outdir, categorical_covariate_keys=['batch'])
    composer.run_pipeline()
    return composer


def get_latent_representation(pianist, adata, **kwargs):
    return pianist.get_latent_representation(adata, **kwargs)


def get_counterfactual(pianist, adata, **kwargs):
    return pianist.get_counterfactual(adata, covariates='marginal', **kwargs)


QUERIES = [get_latent_representation, get_counterfactual]


@pytest.mark.parametrize('query', QUERIES)
@pytest.mark.parametrize('memory_mode', ['CPU', 'SparseCPU', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)])
def test_zero_cell_query_raises(pianist, query, memory_mode):
    with pytest.raises(ValueError, match=r'adata query has no cells \(n_obs is 0\)'):
        query(pianist, make_adata(0, seed=1), memory_mode=memory_mode)


@pytest.mark.parametrize('query', QUERIES)
def test_zero_cell_backed_query_raises(backed_pianist, query, tmp_path):
    with pytest.raises(ValueError, match=r'adata query has no cells \(n_obs is 0\)'):
        query(backed_pianist, read_backed(make_adata(0, seed=1), tmp_path / 'query.h5ad'))


# ---------------------------------------------------------------------------
# Coverage: a query with one cell still works
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('query', QUERIES)
def test_one_cell_query_returns_finite_values(pianist, query):
    result = query(pianist, make_adata(1, seed=1))
    assert result.shape[0] == 1 and np.isfinite(result).all()
