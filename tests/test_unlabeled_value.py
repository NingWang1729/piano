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

UNLABELED = 'NA'  # A custom unlabeled value, instead of the default 'Unknown'
BATCHES = ['a', 'b', UNLABELED, 'Unknown']  # 'Unknown' is an ordinary category when the unlabeled value is 'NA'
N_CELLS = 40
N_GENES = 20
NO_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
MEMORY_MODES = ['CPU', 'SparseCPU', pytest.param('GPU', marks=NO_GPU), pytest.param('SparseGPU', marks=NO_GPU)]


# ---------------------------------------------------------------------------
# Helpers: a small count matrix whose batch covariate holds the unlabeled value 'NA', and a Composer using 'NA'
# ---------------------------------------------------------------------------
def make_adata():
    # Every batch value appears on N_CELLS / 4 cells, in a fixed order
    batches = [BATCHES[i % len(BATCHES)] for i in range(N_CELLS)]
    obs = pd.DataFrame({'batch': pd.Categorical(batches)}, index=[f'cell_{i}' for i in range(N_CELLS)])
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(adata, memory_mode, outdir):
    # No training is needed: initialize_features sets the encodings that the datasets use
    composer = Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        unlabeled=UNLABELED, batch_size=16, deterministic=False, outdir=str(outdir),
    )
    composer.initialize_features()
    return composer


def write_backed_adata(directory):
    # Backed mode reads counts from an .h5ad file, written to pytest's temporary folder
    path = directory / 'adata.h5ad'
    make_adata().write_h5ad(path)
    return ad.read_h5ad(path, backed='r')


def assert_unlabeled_cells_are_zeros(adataset):
    # The batch covariate is the only one, so aug_data holds its one-hot columns: a, b and Unknown (sorted)
    aug_data = adataset.aug_data.cpu().numpy()
    batches = adataset.obs['batch'].astype(str).to_numpy()
    assert aug_data.shape == (N_CELLS, 3)
    assert (aug_data[batches == UNLABELED] == 0).all()
    for column, label in enumerate(['Unknown', 'a', 'b']):
        assert (aug_data[batches == label, column] == 1).all()
    assert (aug_data[batches != UNLABELED].sum(axis=1) == 1).all()


@pytest.fixture(scope='module')
def composer(tmp_path_factory):
    # Shared across tests: the Composer is never trained, so every call builds a new dataset
    return make_composer(make_adata(), 'CPU', tmp_path_factory.mktemp('piano'))


# ---------------------------------------------------------------------------
# The datasets use the Composer's unlabeled value (they used to default to 'Unknown')
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_training_dataset_uses_composer_unlabeled(composer, memory_mode):
    adataset = composer._get_training_adataset(memory_mode)
    assert adataset.unlabeled == UNLABELED


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_query_dataset_uses_composer_unlabeled(composer, memory_mode):
    adataset = composer._get_adataset(make_adata(), memory_mode)
    assert adataset.unlabeled == UNLABELED


def test_backed_datasets_use_composer_unlabeled(tmp_path):
    composer = make_composer(write_backed_adata(tmp_path), 'backed', tmp_path)
    assert composer._get_training_adataset('backed').unlabeled == UNLABELED

    query_directory = tmp_path / 'query'
    query_directory.mkdir()
    assert composer._get_adataset(write_backed_adata(query_directory), 'backed').unlabeled == UNLABELED


# ---------------------------------------------------------------------------
# Coverage: cells labelled 'NA' are encoded as all zeros, and 'Unknown' is an ordinary category
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_unlabeled_cells_encoded_as_zeros(composer, memory_mode):
    assert_unlabeled_cells_are_zeros(composer._get_training_adataset(memory_mode))
    assert_unlabeled_cells_are_zeros(composer._get_adataset(make_adata(), memory_mode))


def test_unlabeled_cells_encoded_as_zeros_backed(tmp_path):
    composer = make_composer(write_backed_adata(tmp_path), 'backed', tmp_path)
    assert_unlabeled_cells_are_zeros(composer._get_training_adataset('backed'))

    query_directory = tmp_path / 'query'
    query_directory.mkdir()
    assert_unlabeled_cells_are_zeros(composer._get_adataset(write_backed_adata(query_directory), 'backed'))
