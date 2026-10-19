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
SMALL_MODEL = {'n_hidden': 16, 'n_layers': 1, 'latent_size': 8}  # Fast to train; the query checks are what is tested


# ---------------------------------------------------------------------------
# Helpers: small count matrix with a batch covariate, optionally with some invalid values, and a model trained on it once
# ---------------------------------------------------------------------------
def make_adata(invalid_values=(), sparse=True, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    X[0, :len(invalid_values)] = invalid_values
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(X) if sparse else X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.fixture(scope='module')
def pianist(tmp_path_factory):
    """A model trained for one CPU epoch on valid counts."""
    composer = Composer(
        make_adata(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False,
        outdir=str(tmp_path_factory.mktemp('query_checks')), **SMALL_MODEL,
    )
    composer.run_pipeline()
    return composer


def get_latent_representation(pianist, adata, **kwargs):
    return pianist.get_latent_representation(adata, **kwargs)


def get_counterfactual(pianist, adata, **kwargs):
    return pianist.get_counterfactual(adata, **kwargs)


QUERIES = [get_latent_representation, get_counterfactual]


# ---------------------------------------------------------------------------
# Invalid query counts raise instead of being changed (NaN stored as 0, -5 as 65531 in uint16)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('query', QUERIES)
@pytest.mark.parametrize('sparse', [True, False], ids=['csr', 'dense'])
@pytest.mark.parametrize('memory_mode', ['CPU', 'SparseCPU'])
@pytest.mark.parametrize('invalid_value', [np.nan, np.inf, -5.0])
def test_invalid_query_counts_raise(pianist, query, sparse, memory_mode, invalid_value):
    with pytest.raises(ValueError, match=r'adata query\.X has .* must be finite and non-negative'):
        query(pianist, make_adata([invalid_value], sparse=sparse, seed=1), memory_mode=memory_mode)


@pytest.mark.parametrize('query', QUERIES)
def test_valid_query_returns_one_row_per_cell(pianist, query):
    result = query(pianist, make_adata(seed=1))
    assert result.shape[0] == N_CELLS and np.isfinite(result).all()


# ---------------------------------------------------------------------------
# The backed or in-memory check uses the memory mode requested for the query, not the training one
# ---------------------------------------------------------------------------
@pytest.fixture
def backed_query(tmp_path):
    make_adata(seed=1).write_h5ad(tmp_path / 'query.h5ad')
    return ad.read_h5ad(tmp_path / 'query.h5ad', backed='r')


@pytest.mark.parametrize('query', QUERIES)
@pytest.mark.parametrize('memory_mode', [None, 'CPU', 'SparseCPU'])
def test_backed_query_refused_in_memory_modes(pianist, query, backed_query, memory_mode):
    expected_mode = memory_mode or pianist.memory_mode
    with pytest.raises(ValueError, match=f"adata query keeps its counts on disk, which memory_mode='{expected_mode}'"):
        query(pianist, backed_query, memory_mode=memory_mode)


@pytest.mark.parametrize('query', QUERIES)
def test_in_memory_query_refused_in_backed_mode(pianist, query):
    with pytest.raises(ValueError, match='adata query is in memory, but backed mode keeps counts on disk'):
        query(pianist, make_adata(seed=1), memory_mode='backed')


def test_backed_query_in_backed_mode_matches_in_memory(pianist, backed_query):
    # A model trained in memory embeds a backed query when backed mode is requested. The latent is a posterior sample,
    # so torch is seeded before each call
    torch.manual_seed(0)
    latent = pianist.get_latent_representation(backed_query, memory_mode='backed')
    torch.manual_seed(0)
    np.testing.assert_allclose(latent, pianist.get_latent_representation(make_adata(seed=1)), rtol=1e-5, atol=1e-6)

