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
BATCHES = ['a', 'b', 'c']


# ---------------------------------------------------------------------------
# Helper: small count matrix with categorical, continuous and sparse continuous covariates
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({
        'batch': rng.choice(BATCHES, N_CELLS),
        'age': rng.normal(size=N_CELLS),
        'drug': rng.choice(['A', 'B'], N_CELLS),
        'dose': rng.uniform(size=N_CELLS),
    }, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(X), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.fixture(scope='module')
def pianist(tmp_path_factory):
    # One epoch in SparseCPU mode, the only mode with sparse continuous covariates; not deterministic, which needs
    # CUBLAS_WORKSPACE_CONFIG on GPU nodes and would stay on for later tests (review item 16)
    pianist = Composer(
        make_adata(), memory_mode='SparseCPU', compile_model=False, n_top_genes=-1,
        categorical_covariate_keys=['batch'], continuous_covariate_keys=['age'],
        sparse_continuous_covariate_keys=[('drug', 'dose')],
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False,
        outdir=str(tmp_path_factory.mktemp('counterfactual')),
    )
    pianist.run_pipeline()
    return pianist


def batch_block(pianist, covariates):
    # Mask and counterfactual values of the batch covariate, which comes first in the covariates
    mask, counterfactuals = pianist._encode_counterfactual_covariates(covariates)
    return mask[:len(BATCHES)], counterfactuals[:len(BATCHES)]


def test_categorical_counterfactual_encoding(pianist):
    # One value is one-hot, k values get 1/k each, 'marginal' gives 1/n to all n values
    mask, counterfactuals = batch_block(pianist, {'batch': 'b'})
    np.testing.assert_array_equal(mask, [0, 0, 0])
    np.testing.assert_array_equal(counterfactuals, [0, 1, 0])
    np.testing.assert_array_equal(batch_block(pianist, {'batch': ['a', 'c', 'a']})[1], np.float32([0.5, 0, 0.5]))
    np.testing.assert_array_equal(batch_block(pianist, {'batch': 'marginal'})[1], np.full(3, 1 / 3, dtype=np.float32))

    # Unchanged covariates keep their original values
    mask, counterfactuals = batch_block(pianist, {})
    np.testing.assert_array_equal(mask, [1, 1, 1])
    np.testing.assert_array_equal(counterfactuals, [0, 0, 0])

    with pytest.raises(ValueError, match="Invalid value 'd' for covariate 'batch'"):
        pianist._encode_counterfactual_covariates({'batch': 'd'})
