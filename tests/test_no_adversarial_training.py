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


# ---------------------------------------------------------------------------
# Helper: small count matrix with a batch covariate
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.mark.parametrize('parameterization', ['ksi-psi', 'mu-theta'])
def test_training_without_adversarial_loss(parameterization, tmp_path):
    # Without adversarial training the adversarial loss is a zero tensor, so the training step can detach it like the others
    pianist = Composer(
        make_adata(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        adversarial=False, parameterization=parameterization, max_epochs=1, batch_size=64, early_stopping=False,
        deterministic=False, outdir=str(tmp_path),
    )
    pianist.run_pipeline()

    losses = pianist.model.training_step(next(iter(pianist.train_adata_loader)), kld_weight=0.0, adv_weight=1.0)
    assert torch.is_tensor(losses['adv']) and losses['adv'].item() == 0
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model.latent_size)
