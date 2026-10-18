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

from piano import Composer, highly_variable_genes
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


def test_backed_var_subset_reads_genes_in_requested_order(files):
    adata, paths = files
    genes = adata.var_names[::-1][:10]  # Not in file order
    dataset = BackedAnnDataset(paths['csr'])
    dataset.set_var_subset(genes)
    rows = torch.arange(5)
    assert torch.equal(dataset[rows], torch.from_numpy(adata[:5, genes].X.toarray()))
    with pytest.raises(ValueError, match='missing'):
        dataset.set_var_subset(['not_a_gene'])


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
# Seurat v3 HVGs on backed data, read from disk in chunks, equal the in-memory result
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('source', ['csr', 'dense'])
@pytest.mark.parametrize('batch_key', ['species', None], ids=['batch_key', 'no_batch_key'])
@pytest.mark.parametrize('chunk_size', [10_000, 97, 299], ids=['one_chunk', 'chunks', 'one_cell_remainder'])
def test_backed_hvgs_match_in_memory(files, source, batch_key, chunk_size):
    adata, paths = files
    expected = highly_variable_genes(adata.copy(), n_top_genes=N_TOP_GENES, batch_key=batch_key, inplace=False)
    backed = highly_variable_genes(ad.read_h5ad(paths[source], backed='r'), n_top_genes=N_TOP_GENES, batch_key=batch_key,
                                   inplace=False, chunk_size=chunk_size)
    np.testing.assert_array_equal(backed['highly_variable'].to_numpy(), expected['highly_variable'].to_numpy())
    for column in ['means', 'variances', 'variances_norm']:
        np.testing.assert_allclose(backed[column].to_numpy(dtype=np.float64), expected[column].to_numpy(dtype=np.float64),
                                   rtol=1e-10, err_msg=column)
    # Chunked sums differ in the last bits, which can swap the ranks of genes tied to that precision
    np.testing.assert_allclose(backed['highly_variable_rank'].to_numpy(dtype=np.float64),
                               expected['highly_variable_rank'].to_numpy(dtype=np.float64), atol=1, equal_nan=True)


def test_hvgs_refuse_to_modify_backed_data_in_place(files):
    _, paths = files
    with pytest.raises(ValueError, match='inplace=False'):
        highly_variable_genes(ad.read_h5ad(paths['csr'], backed='r'), n_top_genes=N_TOP_GENES)


def test_backed_composer_selects_genes_like_in_memory(files):
    adata, paths = files
    in_memory = Composer(adata.copy(), memory_mode='CPU', compile_model=False, n_top_genes=N_TOP_GENES, hvg_batch_key='species', deterministic=False)
    backed = Composer(paths['csr'], memory_mode='backed', compile_model=False, n_top_genes=N_TOP_GENES, hvg_batch_key='species', deterministic=False)
    in_memory.initialize_features()
    backed.initialize_features()
    np.testing.assert_array_equal(np.asarray(backed.var_names), np.asarray(in_memory.var_names))


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
