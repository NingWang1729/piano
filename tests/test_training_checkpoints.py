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
import torch
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 400
N_GENES = 60


# ---------------------------------------------------------------------------
# Helper: negative binomial counts with batch and donor covariates
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS)),
        'donor': pd.Categorical(rng.choice(['d1', 'd2'], N_CELLS)),
    }, index=[f'cell_{i}' for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.negative_binomial(2, 0.3, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(adata, outdir, **kwargs):
    return Composer(adata, memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch', 'donor'], n_top_genes=-1,
                    batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir), **kwargs)


def test_model_after_training_is_the_saved_best_model(tmp_path, capsys):
    pianist = make_composer(make_adata(), tmp_path, max_epochs=4)
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model(**pianist.model_kwargs)

    # Make every epoch after the first look worse in the loss that picks the best epoch ('ref'), so the best epoch is 0, not
    # the last; training itself follows the 'total' loss and is unchanged. Epochs are counted where Composer ends each one
    # (_print_epoch_losses), so this does not depend on the number of cells or the batch size
    finished_epochs = [0]
    print_epoch_losses = pianist._print_epoch_losses
    def count_finished_epochs(*args, **kwargs):
        finished_epochs[0] += 1
        return print_epoch_losses(*args, **kwargs)
    pianist._print_epoch_losses = count_finished_epochs

    training_step = pianist.model.training_step
    def training_step_with_later_epochs_worse(batch, kld_weight, adv_weight):
        losses = training_step(batch, kld_weight, adv_weight)
        if finished_epochs[0] > 0:
            losses['ref'] = losses['ref'] + 1e6
        return losses
    pianist.model.training_step = training_step_with_later_epochs_worse

    pianist.train()
    assert 'Best model at epoch 0 ' in capsys.readouterr().out
    saved = torch.load(f'{tmp_path}/checkpoints/model_checkpoint.pt', weights_only=True)
    # Previously the model kept the last epoch's weights while the checkpoint held the best epoch's
    for name, weights in pianist.model.state_dict().items():
        assert torch.equal(weights.cpu(), saved[name].cpu()), name
