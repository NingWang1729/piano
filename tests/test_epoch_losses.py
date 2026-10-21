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


import math

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50
BATCH_SIZE = 64
MAX_EPOCHS = 2


# ---------------------------------------------------------------------------
# Helper: small counts with a batch covariate
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.mark.parametrize('memory_mode', ['CPU', 'SparseCPU'])
def test_epoch_losses_are_per_cell_when_last_batch_drops_a_key(memory_mode, tmp_path):
    # drop_last=False, so every cell is seen once per epoch and the epoch losses are divided by N_CELLS
    pianist = Composer(
        make_adata(), memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, drop_last=False, early_stopping=False, deterministic=False,
        outdir=str(tmp_path),  # train() writes model checkpoints, so keep them in tmp_path
    )
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model(**pianist.model_kwargs)

    # Record each batch's losses, and leave 'kld' (non-zero, unlike 'adv' without an adversary)
    # out of the last batch of every epoch, as a training step that returns fewer keys would.
    # No current model leaves a key out, so the training step is patched to do it
    recorded = [[] for _ in range(MAX_EPOCHS)]
    n_calls = [0]
    training_step = pianist.model.training_step
    def training_step_without_kld_on_last_batch(batch, kld_weight, adv_weight):
        losses = training_step(batch, kld_weight, adv_weight)
        n_batches = len(pianist.train_adata_loader)  # The loader is built by train(), before the first step
        epoch_idx, batch_idx = divmod(n_calls[0], n_batches)
        n_calls[0] += 1
        if batch_idx == n_batches - 1:
            del losses['kld']
        recorded[epoch_idx].append({k: float(v.detach()) for k, v in losses.items()})
        return losses
    pianist.model.training_step = training_step_without_kld_on_last_batch

    # Keep a copy of the epoch losses that are printed and used to pick the best epoch
    printed = []
    print_epoch_losses = pianist._print_epoch_losses
    def print_and_keep_epoch_losses(epoch_losses, kld_weight, adv_weight, flush=False):
        printed.append({k: float(v) for k, v in epoch_losses.items()})
        return print_epoch_losses(epoch_losses, kld_weight, adv_weight, flush=flush)
    pianist._print_epoch_losses = print_and_keep_epoch_losses

    pianist.train()

    assert n_calls[0] == MAX_EPOCHS * math.ceil(N_CELLS / BATCH_SIZE)
    assert all('kld' not in recorded[epoch_idx][-1] for epoch_idx in range(MAX_EPOCHS))
    assert len(printed) == MAX_EPOCHS
    for epoch_idx in range(MAX_EPOCHS):
        # Every key, including 'kld' that the last batch left out, is a sum over batches divided by the cells
        # (before: 'kld' stayed the unscaled sum over the epoch's batches)
        for k in pianist.LOSS_KEYS:
            expected = sum(losses.get(k, 0.0) for losses in recorded[epoch_idx]) / N_CELLS
            assert printed[epoch_idx][k] == pytest.approx(expected, rel=1e-4, abs=1e-6), k
