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

from piano import Composer, highly_variable_genes

N_CELLS = 200
N_GENES = 60
N_TOP_GENES = 20
SMALL_MODEL = {'n_hidden': 16, 'n_layers': 1, 'latent_size': 8}  # Fast to train; gene selection is what is tested
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
# Backed mode is left out: backed AnnDatas are read-only and backed mode already selects with inplace=False
# (its gene order against in-memory mode is covered by test_backed_composer_selects_genes_like_in_memory)
IN_MEMORY_MODES = ['CPU', 'SparseCPU', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)]


# ---------------------------------------------------------------------------
# Helpers: negative binomial counts with a batch covariate, and a small Composer
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    mu = 10 ** rng.uniform(-1.5, 1, N_GENES) * rng.uniform(0.5, 2.0, (N_CELLS, N_GENES))
    counts = rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[f'cell_{seed}_{i}' for i in range(N_CELLS)])
    return ad.AnnData(X=csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f'gene_{i}' for i in range(N_GENES)]))


def make_composer(adata, outdir, memory_mode='CPU', **kwargs):
    # outdir=tmp_path keeps the Composer's checkpoints out of the repo
    return Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'],
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir), **SMALL_MODEL, **kwargs,
    )


@pytest.mark.parametrize('memory_mode', IN_MEMORY_MODES)
def test_hvg_selection_leaves_callers_adata_unchanged(memory_mode, tmp_path):
    # Previously HVG selection subset the caller's AnnData in place (60 -> 20 genes) and added var columns
    adata = make_adata()
    original = adata.copy()
    pianist = make_composer(adata, tmp_path, memory_mode, n_top_genes=N_TOP_GENES, hvg_batch_key='batch')
    pianist.run_pipeline()

    assert adata.shape == original.shape
    assert adata.var.equals(original.var)
    assert adata.uns.keys() == original.uns.keys()
    assert (adata.X != original.X).nnz == 0

    # The Composer still trains on the HVGs, in the AnnData's (var) gene order, not a sorted order
    expected = highly_variable_genes(original, n_top_genes=N_TOP_GENES, batch_key='batch', inplace=False)
    assert list(pianist.var_names) == list(expected.index[expected['highly_variable'].to_numpy()])
    assert pianist.get_latent_representation().shape == (N_CELLS, SMALL_MODEL['latent_size'])


def test_hvg_selection_leaves_every_callers_adata_unchanged(tmp_path):
    # With several AnnDatas, HVG selection previously subset every one of them in place
    adatas = [make_adata(0), make_adata(1)]
    originals = [adata.copy() for adata in adatas]
    pianist = make_composer(adatas, tmp_path, n_top_genes=N_TOP_GENES, hvg_batch_key='batch')
    pianist.run_pipeline()

    for adata, original in zip(adatas, originals):
        assert adata.shape == original.shape
        assert adata.var.equals(original.var)
        assert adata.uns.keys() == original.uns.keys()
        assert (adata.X != original.X).nnz == 0

    # HVGs are selected across both AnnDatas, in the AnnData's (var) gene order
    expected = highly_variable_genes(originals, n_top_genes=N_TOP_GENES, batch_key='batch', inplace=False)
    assert list(pianist.var_names) == list(expected.index[expected['highly_variable'].to_numpy()])
