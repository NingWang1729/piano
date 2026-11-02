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
import warnings

import numpy as np

# Composer settings in scVI mode (Composer(mode='scvi')) when not passed: those of scvi.model.SCVI and its train() in
# scvi-tools 1.4.2, except the architecture, the batch size and the matmul precision (larger and faster, with equivalent
# results). max_epochs is scvi_max_epochs(number of cells); n_top_genes=-1 keeps every gene, as scVI does not select genes
# itself
SCVI_DEFAULTS = {
    # 4 hidden layers of 512 units, larger than scvi-tools' default of 1 x 128, which is rarely used as is
    'n_hidden': 512,
    'n_layers': 4,
    'latent_size': 10,
    'adversarial': False,
    'batch_size': 512,  # scvi-tools: 128
    'lr': 1e-3,
    'weight_decay': 1e-6,
    'max_kld_weight': 1.0,
    'n_annealing_epochs': 400,
    'drop_last': False,
    'early_stopping': False,
    'min_delta': 0.0,
    'patience': 45,
    'n_top_genes': -1,
    'matmul_precision': 'high',  # TF32 on Ampere and newer GPUs; scvi-tools trains in FP32
}

# scVI trains on this fraction of the cells and keeps the rest for validation
SCVI_TRAIN_FRACTION = 0.9

# Adam's epsilon in scVI's optimizer; large next to PyTorch's 1e-8, it damps updates of parameters with small gradients
SCVI_ADAM_EPS = 0.01


def scvi_max_epochs(n_cells, epochs_cap=400, decay_at_n_cells=20_000):
    """
    scVI's default number of epochs: epochs_cap up to decay_at_n_cells cells, then fewer in proportion to the number
    of cells (rounded), at least 1, so that larger datasets train for a similar number of steps. Warns if it is 1.
    """
    max_epochs = max(min(round(decay_at_n_cells / n_cells * epochs_cap), epochs_cap), 1)
    if max_epochs == 1:
        warnings.warn(
            f"The default number of epochs is 1 for {n_cells} cells: pass max_epochs to train for longer",
            UserWarning, stacklevel=2,
        )
    return max_epochs


def scvi_split_sizes(n_cells, batch_size, drop_last):
    """
    Numbers of training and validation cells, as scVI's default split: SCVI_TRAIN_FRACTION of the cells (rounded up)
    train and the others validate. Without drop_last, a last training batch of 1 or 2 cells, too small for batch
    normalisation, moves to the validation set (or, if there is none, is left out).
    """
    n_train = math.ceil(SCVI_TRAIN_FRACTION * n_cells)
    n_validation = n_cells - n_train
    n_last_batch = n_train % batch_size
    if not drop_last and 0 < n_last_batch < 3:
        n_train -= n_last_batch
        if n_validation > 0:
            n_validation += n_last_batch
    return n_train, n_validation


def scvi_split(n_cells, batch_size, drop_last, random_seed):
    """
    Training and validation cell indices: a random permutation of the cells (numpy RandomState with random_seed,
    None for unseeded), whose first cells validate and the next train, in scvi_split_sizes' numbers.
    """
    n_train, n_validation = scvi_split_sizes(n_cells, batch_size, drop_last)
    order = np.random.RandomState(seed=random_seed).permutation(n_cells)
    return order[n_validation:n_validation + n_train], order[:n_validation]


def scvi_kl_weight(epoch, n_annealing_epochs, min_kl_weight, max_kl_weight):
    """
    scVI's KL weight in an epoch: linear in the epoch index from min_kl_weight (epoch 0) towards max_kl_weight over
    n_annealing_epochs epochs, constant within an epoch, then max_kl_weight. With 400 annealing epochs (the default), a
    200-epoch run ends near half of max_kl_weight.
    """
    if epoch >= n_annealing_epochs:
        return max_kl_weight
    return min_kl_weight + (max_kl_weight - min_kl_weight) * epoch / n_annealing_epochs
