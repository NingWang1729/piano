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

from piano.utils.io import read_adata


# ---------------------------------------------------------------------------
# Helper: small count matrix with a categorical covariate
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], 40))}, index=[f'cell_{i}' for i in range(40)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(40, 12)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(12)]
    return adata


def assert_same_data(read, expected):
    assert read.obs_names.equals(expected.obs_names) and read.var_names.equals(expected.var_names)
    np.testing.assert_array_equal(read.X.toarray(), expected.X.toarray())
    assert read.obs['batch'].astype(str).equals(expected.obs['batch'].astype(str))


@pytest.mark.parametrize('suffix', ['.h5ad', '.zarr', '.H5AD'])
def test_read_adata_reads_each_format(tmp_path, suffix):
    adata = make_adata()
    path = tmp_path / f'data{suffix}'
    adata.write_zarr(path) if suffix == '.zarr' else adata.write_h5ad(path)
    assert_same_data(read_adata(str(path)), adata)


def test_read_adata_refuses_other_formats(tmp_path):
    with pytest.raises(ValueError, match='reads .h5ad and .zarr files'):
        read_adata(str(tmp_path / 'data.csv'))
