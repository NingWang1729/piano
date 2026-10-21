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

N_CELLS = (200, 100)  # Cells of the two training AnnDatas, in the order passed
N_GENES = 50
TRAINING_MODE = 'CPU'
NO_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
MEMORY_MODES = [None, 'CPU', 'SparseCPU', pytest.param('GPU', marks=NO_GPU), pytest.param('SparseGPU', marks=NO_GPU)]


# ---------------------------------------------------------------------------
# Helpers: small count matrices with a batch covariate, and a Composer trained for one epoch on two of them
# ---------------------------------------------------------------------------
def make_adata(n_cells, seed):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], n_cells))}, index=[f'{seed}_{i}' for i in range(n_cells)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(n_cells, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def train_composer(adatas, memory_mode, outdir):
    pianist = Composer(
        adatas, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir),
    )
    pianist.run_pipeline()
    return pianist


@pytest.fixture(scope='module')
def pianist(tmp_path_factory):
    adatas = [make_adata(n_cells, seed) for seed, n_cells in enumerate(N_CELLS)]
    return train_composer(adatas, TRAINING_MODE, tmp_path_factory.mktemp('piano'))


@pytest.fixture(scope='module')
def backed_pianist(tmp_path_factory):
    # Backed mode reads counts from .h5ad files, written to pytest's temporary folder
    directory = tmp_path_factory.mktemp('backed')
    adatas = []
    for seed, n_cells in enumerate(N_CELLS):
        path = directory / f'adata_{seed}.h5ad'
        make_adata(n_cells, seed).write_h5ad(path)
        adatas.append(ad.read_h5ad(path, backed='r'))
    return train_composer(adatas, 'backed', directory)


# ---------------------------------------------------------------------------
# adata=None: every cell of every training AnnData, in every memory mode
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_latents_of_all_training_adatas(pianist, memory_mode):
    # Previously memory modes other than the training mode gave only the first AnnData's 200 cells
    latent = pianist.get_latent_representation(memory_mode=memory_mode)
    assert latent.shape[0] == sum(N_CELLS)
    assert np.allclose(latent, pianist.get_latent_representation(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_counterfactuals_of_all_training_adatas(pianist, memory_mode):
    # Previously memory modes other than the training mode gave only the first AnnData's 200 cells
    counterfactuals = pianist.get_counterfactual(memory_mode=memory_mode)
    assert counterfactuals.shape == (sum(N_CELLS), N_GENES)
    assert np.allclose(counterfactuals, pianist.get_counterfactual(), rtol=1e-4, atol=1e-5)


def test_rows_follow_training_order(pianist):
    # Coverage: the rows are the first AnnData's cells, then the second's. Copies are passed, so each gives its own cells
    each = [pianist.get_latent_representation(adata.copy()) for adata in pianist.adata]
    assert np.allclose(pianist.get_latent_representation(), np.concatenate(each), rtol=1e-4, atol=1e-5)


def test_backed_latents_of_all_training_adatas(backed_pianist):
    # Coverage: backed mode, the one memory mode the in-memory Composer above cannot use
    assert backed_pianist.get_latent_representation().shape[0] == sum(N_CELLS)


# ---------------------------------------------------------------------------
# An explicitly passed AnnData gives exactly its own cells
# ---------------------------------------------------------------------------
def test_first_training_adata_gives_its_own_cells(pianist):
    # Previously the Composer's own first AnnData was mistaken for all the training data and gave all 300 cells
    assert pianist.get_latent_representation(pianist.adata[0]).shape[0] == N_CELLS[0]


# ---------------------------------------------------------------------------
# A memory mode the training AnnDatas cannot use raises, naming the training AnnData
# ---------------------------------------------------------------------------
def test_backed_mode_with_in_memory_training_adatas_raises(pianist):
    # Previously the error named the AnnData 'query' rather than the training AnnData's index
    with pytest.raises(ValueError, match='adata 0 is in memory'):
        pianist.get_latent_representation(memory_mode='backed')


def test_in_memory_mode_with_backed_training_adatas_raises(backed_pianist):
    # Previously the error named the AnnData 'query' rather than the training AnnData's index
    with pytest.raises(ValueError, match='adata 0 keeps its counts on disk'):
        backed_pianist.get_latent_representation(memory_mode='CPU')
