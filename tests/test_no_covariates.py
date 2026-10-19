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

from piano import Composer

N_CELLS = 200
N_GENES = 50
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')


# ---------------------------------------------------------------------------
# Helper: small count matrix whose obs has no columns, so there are no covariates to encode
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=[str(i) for i in range(N_CELLS)]))
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.mark.parametrize('memory_mode', [
    'CPU',
    'SparseCPU',
    pytest.param('GPU', marks=needs_cuda),
    pytest.param('SparseGPU', marks=needs_cuda),
])
def test_training_without_covariates(memory_mode, tmp_path):
    # Sparse modes used to fail at dataset creation with "RuntimeError: hstack expects a non-empty TensorList"
    pianist = Composer(
        make_adata(), memory_mode=memory_mode, compile_model=False, adversarial=False, n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(tmp_path),
    )
    pianist.run_pipeline()

    # Each batch holds only the genes, and inference works end to end
    assert pianist.train_adataset[torch.tensor([0, 1])].shape == (2, N_GENES)
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model.latent_size)
    assert pianist.get_counterfactual().shape == (N_CELLS, N_GENES)
