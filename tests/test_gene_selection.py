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

from piano import Composer, highly_variable_genes

N_CELLS = 200
N_GENES = 60
N_TOP_GENES = 20
SMALL_MODEL = {'n_hidden': 16, 'n_layers': 1, 'latent_size': 8}  # Fast to train; gene selection is what is tested
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
# Backed mode is left out: backed AnnDatas are read-only and backed mode already selects with inplace=False
# (its gene order against in-memory mode is covered by test_backed_composer_selects_genes_like_in_memory)
IN_MEMORY_MODES = ['CPU', 'SparseCPU', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)]


# ---------------------------------------------------------------------------
# Helpers: negative binomial counts with a batch covariate, and a small Composer
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    mu = 10 ** rng.uniform(-1.5, 1, N_GENES) * rng.uniform(0.5, 2.0, (N_CELLS, N_GENES))
    counts = rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[f'cell_{seed}_{i}' for i in range(N_CELLS)])
    return ad.AnnData(X=csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f'gene_{i}' for i in range(N_GENES)]))


def make_composer(adata, outdir, memory_mode='CPU', **kwargs):
    # outdir=tmp_path keeps the Composer's checkpoints out of the repo
    return Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'],
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir), **SMALL_MODEL, **kwargs,
    )


@pytest.mark.parametrize('memory_mode', IN_MEMORY_MODES)
def test_hvg_selection_leaves_callers_adata_unchanged(memory_mode, tmp_path):
    # Previously HVG selection subset the caller's AnnData in place (60 -> 20 genes) and added var columns
    adata = make_adata()
    original = adata.copy()
    pianist = make_composer(adata, tmp_path, memory_mode, n_top_genes=N_TOP_GENES, hvg_batch_key='batch')
    pianist.run_pipeline()

    assert adata.shape == original.shape
    assert adata.var.equals(original.var)
    assert adata.uns.keys() == original.uns.keys()
    assert (adata.X != original.X).nnz == 0

    # The Composer still trains on the HVGs, in the AnnData's (var) gene order, not a sorted order
    expected = highly_variable_genes(original, n_top_genes=N_TOP_GENES, batch_key='batch', inplace=False)
    assert list(pianist.var_names) == list(expected.index[expected['highly_variable'].to_numpy()])
    assert pianist.get_latent_representation().shape == (N_CELLS, SMALL_MODEL['latent_size'])


def test_hvg_selection_leaves_every_callers_adata_unchanged(tmp_path):
    # With several AnnDatas, HVG selection previously subset every one of them in place
    adatas = [make_adata(0), make_adata(1)]
    originals = [adata.copy() for adata in adatas]
    pianist = make_composer(adatas, tmp_path, n_top_genes=N_TOP_GENES, hvg_batch_key='batch')
    pianist.run_pipeline()

    for adata, original in zip(adatas, originals):
        assert adata.shape == original.shape
        assert adata.var.equals(original.var)
        assert adata.uns.keys() == original.uns.keys()
        assert (adata.X != original.X).nnz == 0

    # HVGs are selected across both AnnDatas, in the AnnData's (var) gene order
    expected = highly_variable_genes(originals, n_top_genes=N_TOP_GENES, batch_key='batch', inplace=False)
    assert list(pianist.var_names) == list(expected.index[expected['highly_variable'].to_numpy()])


# ---------------------------------------------------------------------------
# Gene sets (geneset_path), in every memory mode
# ---------------------------------------------------------------------------
# Backed mode reads an .h5ad written to the test's tmp_path, so gene sets are tested in every memory mode
ALL_MODES = ['CPU', 'SparseCPU', 'backed', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)]


def make_data(memory_mode, tmp_path, seed=0):
    # Backed mode needs an AnnData read from disk; the .h5ad goes to tmp_path, not next to any data
    adata = make_adata(seed)
    if memory_mode != 'backed':
        return adata
    path = tmp_path / 'adata.h5ad'
    adata.write_h5ad(path)
    return ad.read_h5ad(path, backed='r')


def write_gene_set(tmp_path, genes, name='gene_set.csv'):
    # One gene per line with no header, as the CLIs' gene-set files; written to tmp_path
    path = tmp_path / name
    path.write_text(''.join(f'{gene}\n' for gene in genes))
    return str(path)


@pytest.mark.parametrize('memory_mode', ALL_MODES)
def test_gene_set_genes_in_adata_order_and_saved(memory_mode, tmp_path):
    # Scrambled, and sorted order (gene_15, gene_3, gene_30, gene_42, gene_7) differs from the AnnData's order
    geneset_path = write_gene_set(tmp_path, ['gene_42', 'gene_7', 'gene_15', 'gene_3', 'gene_30'])
    pianist = make_composer(make_data(memory_mode, tmp_path), tmp_path, memory_mode, n_top_genes=N_TOP_GENES, geneset_path=geneset_path)
    # Previously training failed when saving the model: 'numpy.ndarray' object has no attribute 'to_series'
    pianist.run_pipeline()

    # The gene-set genes are used in the AnnData's gene order (previously sorted), as a pandas Index
    expected = ['gene_3', 'gene_7', 'gene_15', 'gene_30', 'gene_42']
    assert isinstance(pianist.var_names, pd.Index)
    assert list(pianist.var_names) == expected
    assert pianist.get_latent_representation().shape == (N_CELLS, SMALL_MODEL['latent_size'])

    # Training saved the genes next to the model checkpoint, and the Composer pickles (both in tmp_path)
    saved = pd.read_csv(f'{pianist.checkpoint_path}/var_names.csv', header=None, dtype=str)
    assert saved[0].tolist() == expected
    pianist.save(tmp_path / 'pianist.pkl')
    assert list(Composer.load(tmp_path / 'pianist.pkl').var_names) == expected


@pytest.mark.parametrize('memory_mode', ['CPU', 'backed'])
def test_gene_set_with_numeric_gene_ids(memory_mode, tmp_path):
    # Numeric gene IDs (e.g. Entrez IDs) are read as strings, so they match string var_names; previously the file was
    # read as integers, which never matched. Every memory mode reads the gene set the same way, so CPU and backed suffice
    adata = make_adata()
    adata.var_names = [str(i) for i in range(N_GENES)]
    if memory_mode == 'backed':
        path = tmp_path / 'adata.h5ad'  # Backed mode reads from disk; written to tmp_path
        adata.write_h5ad(path)
        adata = ad.read_h5ad(path, backed='r')
    geneset_path = write_gene_set(tmp_path, ['42', '7', '15'])
    pianist = make_composer(adata, tmp_path, memory_mode, n_top_genes=N_TOP_GENES, geneset_path=geneset_path)
    pianist.initialize_features()
    assert list(pianist.var_names) == ['7', '15', '42']


@pytest.mark.parametrize('memory_mode', ALL_MODES)
def test_gene_set_used_with_all_genes_requested(memory_mode, tmp_path):
    # n_top_genes=-1 means all genes only without a gene set; previously the gene set was silently ignored
    geneset_path = write_gene_set(tmp_path, ['gene_9', 'gene_2'])
    pianist = make_composer(make_data(memory_mode, tmp_path), tmp_path, memory_mode, n_top_genes=-1, geneset_path=geneset_path)
    pianist.initialize_features()
    assert list(pianist.var_names) == ['gene_2', 'gene_9']
