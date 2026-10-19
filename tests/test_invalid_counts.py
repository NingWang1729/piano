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
# Helper: small count matrix with a batch covariate, optionally with some invalid values
# ---------------------------------------------------------------------------
def make_adata(invalid_values=(), sparse=True, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(dtype)
    X[0, :len(invalid_values)] = invalid_values
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(X) if sparse else X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(adata, outdir, memory_mode='CPU'):
    return Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir),
    )


@pytest.mark.parametrize('sparse', [True, False], ids=['csr', 'dense'])
@pytest.mark.parametrize('memory_mode', ['CPU', 'SparseCPU'])
@pytest.mark.parametrize('invalid_value', [np.nan, np.inf, -np.inf, -1.0, -0.5])
def test_invalid_counts_raise_when_composer_is_created(invalid_value, memory_mode, sparse, tmp_path):
    # Previously these failed at the first batch with an error from torch, or were stored as other counts as uint16
    with pytest.raises(ValueError, match='must be finite and non-negative'):
        make_composer(make_adata([invalid_value], sparse=sparse), tmp_path, memory_mode)


def test_error_reports_how_many_values_are_invalid(tmp_path):
    with pytest.raises(ValueError, match=r'adata 0\.X has 2 NaN, 1 infinite and 1 negative values'):
        make_composer(make_adata([np.nan, np.nan, np.inf, -3.0]), tmp_path)


@pytest.mark.parametrize('dtype', [np.float32, np.int64, np.uint16])
@pytest.mark.parametrize('sparse', [True, False], ids=['csr', 'dense'])
def test_valid_counts_train(sparse, dtype, tmp_path):
    # Valid counts of common dtypes pass the check and train for one epoch
    pianist = make_composer(make_adata(sparse=sparse, dtype=dtype), tmp_path)
    pianist.run_pipeline()
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model.latent_size)
