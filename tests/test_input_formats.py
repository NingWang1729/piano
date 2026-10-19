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
from scipy.sparse import csr_matrix, csc_matrix, csr_array

from piano import Composer

N_CELLS = 200
N_GENES = 50
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
MEMORY_MODES = ['CPU', 'SparseCPU', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)]
SPARSE_FORMATS = [csc_matrix, csr_array]  # Formats besides csr_matrix, the reference


# ---------------------------------------------------------------------------
# Helper: small count matrix with a batch covariate
# ---------------------------------------------------------------------------
def make_adata(to_format, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=to_format(X), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(adata, memory_mode, outdir):
    return Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir),
    )


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
@pytest.mark.parametrize('to_format', SPARSE_FORMATS, ids=lambda f: f.__name__)
def test_sparse_formats_give_the_same_batches_as_csr_matrix(to_format, memory_mode, tmp_path):
    # Every cell's genes and covariates match those from csr_matrix input
    batches = {}
    for name, fmt in [('reference', csr_matrix), ('format', to_format)]:
        pianist = make_composer(make_adata(fmt), memory_mode, tmp_path / name)
        pianist.initialize_features()
        pianist.prepare_data()
        batches[name] = pianist.train_adataset[torch.arange(N_CELLS)].cpu()
    assert torch.equal(batches['format'], batches['reference'])


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
@pytest.mark.parametrize('to_format', SPARSE_FORMATS, ids=lambda f: f.__name__)
def test_sparse_formats_train(to_format, memory_mode, tmp_path):
    pianist = make_composer(make_adata(to_format), memory_mode, tmp_path)
    pianist.run_pipeline()
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model.latent_size)
