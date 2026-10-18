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

import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix

from piano import Composer
from piano.utils.data import AnnDataset, BackedAnnDataset

N_CELLS = 300
N_GENES = 80
N_TOP_GENES = 30
COVARIATES = ['species', 'donor']
SMALL_MODEL = {'n_hidden': 16, 'n_layers': 1, 'latent_size': 8}  # Fast to train; the data path is what is tested


# ---------------------------------------------------------------------------
# Helper: negative binomial counts with a species batch, written as CSR and dense .h5ad once per test module
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    mu = 10 ** rng.uniform(-1.5, 1, N_GENES) * rng.uniform(0.5, 2.0, (N_CELLS, N_GENES))
    counts = rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32)
    obs = pd.DataFrame({
        'species': pd.Categorical(rng.choice(['human', 'macaque', 'marmoset'], N_CELLS)),
        'donor': pd.Categorical(rng.choice(['d1', 'd2'], N_CELLS)),
    }, index=[f'cell_{seed}_{i}' for i in range(N_CELLS)])
    return ad.AnnData(X=csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f'gene_{i}' for i in range(N_GENES)]))


@pytest.fixture(scope='module')
def files(tmp_path_factory):
    """The data in memory, and paths of it written with X stored as CSR and as dense."""
    directory = tmp_path_factory.mktemp('backed')
    adata = make_adata()
    paths = {'csr': str(directory / 'csr.h5ad'), 'dense': str(directory / 'dense.h5ad')}
    adata.write_h5ad(paths['csr'])
    ad.AnnData(X=adata.X.toarray(), obs=adata.obs, var=adata.var).write_h5ad(paths['dense'])
    return adata, paths


def in_memory_batch(adata, rows):
    return AnnDataset(adata.copy(), memory_mode='CPU', categorical_covariate_keys=COVARIATES)[rows]


# ---------------------------------------------------------------------------
# Backed batches read only their cells from disk and equal the in-memory batches
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('source', ['csr', 'dense', 'backed_anndata'])
def test_backed_batches_match_in_memory(files, source):
    adata, paths = files
    data = ad.read_h5ad(paths['csr'], backed='r') if source == 'backed_anndata' else paths[source]
    dataset = BackedAnnDataset(data, categorical_covariate_keys=COVARIATES)
    rows = torch.tensor([7, 3, N_CELLS - 1, 3, 0])  # Unsorted, with a repeat
    # Previously backed batches held only the covariates (no genes)
    assert torch.equal(dataset[rows], in_memory_batch(adata, rows))
    assert dataset.adata.isbacked  # The counts stay on disk


def test_backed_dataset_reopens_its_file_after_pickling(files):
    # DataLoader workers receive a pickled copy, which must not carry the open file
    _, paths = files
    dataset = BackedAnnDataset(paths['csr'], categorical_covariate_keys=COVARIATES)
    rows = torch.tensor([4, 1])
    assert torch.equal(pickle.loads(pickle.dumps(dataset))[rows], dataset[rows])


def test_backed_dataset_refuses_in_memory_anndata(files):
    adata, _ = files
    with pytest.raises(ValueError, match='keeps counts on disk'):
        BackedAnnDataset(adata)


def test_backed_dataset_refuses_csc_on_disk(tmp_path):
    adata = make_adata()
    adata.X = csc_matrix(adata.X)
    adata.write_h5ad(tmp_path / 'csc.h5ad')
    with pytest.raises(ValueError, match='CSC'):
        BackedAnnDataset(str(tmp_path / 'csc.h5ad'))


# ---------------------------------------------------------------------------
# End-to-end: one CPU training epoch in backed mode
# ---------------------------------------------------------------------------
def train_one_epoch(adata, outdir, **kwargs):
    pianist = Composer(
        adata, memory_mode='backed', compile_model=False, categorical_covariate_keys=COVARIATES,
        n_top_genes=N_TOP_GENES, max_epochs=1, batch_size=64, deterministic=False, outdir=str(outdir), **SMALL_MODEL, **kwargs,
    )
    pianist.run_pipeline()
    return pianist.get_latent_representation()


def test_backed_composer_trains_one_epoch(files, tmp_path):
    _, paths = files
    latent = train_one_epoch(paths['csr'], tmp_path)
    assert latent.shape == (N_CELLS, SMALL_MODEL['latent_size']) and np.isfinite(latent).all()
