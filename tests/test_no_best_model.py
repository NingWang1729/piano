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


import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50


# ---------------------------------------------------------------------------
# Helper: small counts with a batch covariate
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.mark.parametrize('max_epochs', [1, 3])
def test_training_without_finite_loss_raises(max_epochs, tmp_path):
    pianist = Composer(
        make_adata(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=max_epochs, batch_size=64, early_stopping=False, deterministic=False, outdir=str(tmp_path),
    )
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model(**pianist.model_kwargs)

    # Make training diverge: the loss that picks the best epoch ('ref') is NaN in every batch
    training_step = pianist.model.training_step
    def training_step_with_nan_loss(batch, kld_weight, adv_weight):
        losses = training_step(batch, kld_weight, adv_weight)
        losses['ref'] = losses['ref'] * float('nan')
        return losses
    pianist.model.training_step = training_step_with_nan_loss

    # Previously None was saved as the checkpoint, then loading it failed with
    # "TypeError: Expected state_dict to be dict-like, got <class 'NoneType'>"
    with pytest.raises(RuntimeError, match='no best model to save'):
        pianist.train()
    assert not os.path.exists(f'{tmp_path}/checkpoints/model_checkpoint.pt')
