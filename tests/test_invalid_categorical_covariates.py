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
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50


# ---------------------------------------------------------------------------
# Helpers: small counts with categorical 'batch' and 'donor' covariates
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS)),
        'donor': rng.choice(['d1', 'd2', 'd3'], N_CELLS),
    }, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(adata, outdir):
    return Composer(
        adata, memory_mode='SparseCPU', compile_model=False, categorical_covariate_keys=['batch', 'donor'],
        n_top_genes=-1, max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir),
    )


QUERY_METHODS = {
    'latent': lambda pianist, query: pianist.get_latent_representation(adata=query),
    'counterfactual': lambda pianist, query: pianist.get_counterfactual(adata=query, covariates='marginal'),
}


# ---------------------------------------------------------------------------
# Missing columns
# ---------------------------------------------------------------------------
def test_missing_categorical_covariate_column_raises_when_composer_is_created(tmp_path):
    # Previously no error at creation; training then failed with a bare AssertionError from encode_categorical_covariates
    second = make_adata(seed=1)
    del second.obs['donor']
    with pytest.raises(ValueError, match=r"adata 1\.obs has no column 'donor', which categorical_covariate_keys names"):
        make_composer([make_adata(), second], tmp_path)


@pytest.fixture(scope='module')
def trained_pianist(tmp_path_factory):
    # One epoch in SparseCPU mode; the model and logs go to a temporary directory from pytest, not next to the data
    pianist = make_composer(make_adata(), tmp_path_factory.mktemp('query'))
    pianist.run_pipeline()
    return pianist


@pytest.mark.parametrize('method', QUERY_METHODS.values(), ids=QUERY_METHODS.keys())
def test_missing_query_categorical_covariate_column_raises(trained_pianist, method):
    # Previously a bare KeyError from building the query dataset
    query = make_adata(seed=2)
    del query.obs['donor']
    with pytest.raises(ValueError, match=r"adata query\.obs has no column 'donor', which categorical_covariate_keys names"):
        method(trained_pianist, query)
