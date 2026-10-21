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


import re
import warnings

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
# Backed mode reads an .h5ad written to the test's tmp_path, so gene sets are tested in every memory mode.
# With a gene set, the default n_top_genes uses the whole set, so tests that are not about n_top_genes leave it out
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
    pianist = make_composer(make_data(memory_mode, tmp_path), tmp_path, memory_mode, geneset_path=geneset_path)
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
    pianist = make_composer(adata, tmp_path, memory_mode, geneset_path=geneset_path)
    pianist.initialize_features()
    assert list(pianist.var_names) == ['7', '15', '42']


@pytest.mark.parametrize('memory_mode', ALL_MODES)
def test_gene_set_used_with_all_genes_requested(memory_mode, tmp_path):
    # n_top_genes=-1 means all genes only without a gene set; previously the gene set was silently ignored
    geneset_path = write_gene_set(tmp_path, ['gene_9', 'gene_2'])
    pianist = make_composer(make_data(memory_mode, tmp_path), tmp_path, memory_mode, n_top_genes=-1, geneset_path=geneset_path)
    pianist.initialize_features()
    assert list(pianist.var_names) == ['gene_2', 'gene_9']


def test_read_gene_set_genes(tmp_path):
    # Coverage of the public helper the Composer and both CLIs use (new, so imported here rather than at the top)
    import piano
    import piano.utils
    from piano import read_gene_set_genes
    assert 'read_gene_set_genes' in piano.__all__ and 'read_gene_set_genes' in piano.utils.__all__

    # Numeric gene IDs are read as strings, so they match string var_names; a repeated gene is used once
    var_names = pd.Index(['1', '2', '3', '10', '20'])
    genes = read_gene_set_genes(var_names, write_gene_set(tmp_path, ['20', '2', '20', '10']))
    assert isinstance(genes, pd.Index)
    assert list(genes) == ['2', '10', '20']  # In var_names order, not the file's or sorted order


@pytest.mark.parametrize('memory_mode', ALL_MODES)
def test_missing_gene_set_genes_warn_at_creation(memory_mode, tmp_path):
    geneset_path = write_gene_set(tmp_path, ['gene_12', 'not_a_gene_1', 'gene_5', 'not_a_gene_2', 'gene_40'])
    adata = make_data(memory_mode, tmp_path)
    # Previously the 2 genes not in the data were dropped silently
    with pytest.warns(UserWarning) as record:
        pianist = make_composer(adata, tmp_path, memory_mode, geneset_path=geneset_path)
    messages = [str(warning.message) for warning in record]
    assert any(re.search(r'\b2\b', message) and 'not_a_gene_1' in message for message in messages), messages

    # The genes that were found are used
    pianist.initialize_features()
    assert list(pianist.var_names) == ['gene_5', 'gene_12', 'gene_40']


@pytest.mark.parametrize('memory_mode', ALL_MODES)
def test_gene_set_with_no_genes_in_data_raises_at_creation(memory_mode, tmp_path):
    # Previously the Composer was created, and training ran on zero genes
    geneset_path = write_gene_set(tmp_path, ['not_a_gene_1', 'not_a_gene_2'])
    adata = make_data(memory_mode, tmp_path)
    with pytest.raises(ValueError, match=re.escape(geneset_path)):
        make_composer(adata, tmp_path, memory_mode, geneset_path=geneset_path)


@pytest.mark.parametrize('memory_mode', ALL_MODES)
def test_empty_gene_set_raises_at_creation(memory_mode, tmp_path):
    # Previously the file was only read when genes were selected, failing there with pandas' EmptyDataError
    geneset_path = write_gene_set(tmp_path, [])
    adata = make_data(memory_mode, tmp_path)
    with pytest.raises(ValueError, match=re.escape(geneset_path)):
        make_composer(adata, tmp_path, memory_mode, geneset_path=geneset_path)


# ---------------------------------------------------------------------------
# n_top_genes with a gene set: that many HVGs from within the set; None (the default) means the whole set
# ---------------------------------------------------------------------------
# 30 of the 60 genes, scrambled (not in the AnnData's order)
GENE_SET_30 = [f'gene_{i}' for i in np.random.default_rng(3).permutation(N_GENES)[:30]]


def in_adata_order(var_names, genes):
    # The genes that are in var_names, in var_names' order
    genes = set(genes)
    return [gene for gene in var_names if gene in genes]


def hvgs_within(adata, genes, n_top_genes, batch_key):
    # The HVGs of an AnnData holding only the gene-set genes, in the AnnData's gene order
    hvg = highly_variable_genes(adata[:, in_adata_order(adata.var_names, genes)].copy(), n_top_genes=n_top_genes, batch_key=batch_key, inplace=False)
    return list(hvg.index[hvg['highly_variable'].to_numpy()])


def n_top_genes_warnings(record):
    # The messages of the warnings about n_top_genes, which name it as n_top_genes=<value> (other UserWarnings, e.g. from
    # anndata or about missing gene-set genes, are left out; the latter name the file, whose tmp_path may hold 'n_top_genes')
    return [str(warning.message) for warning in record if 'n_top_genes=' in str(warning.message)]


@pytest.mark.parametrize('memory_mode', ALL_MODES)
def test_n_top_genes_selected_within_gene_set(memory_mode, tmp_path):
    # Previously the whole gene set (30 genes) was used whatever n_top_genes was
    geneset_path = write_gene_set(tmp_path, GENE_SET_30)
    pianist = make_composer(make_data(memory_mode, tmp_path), tmp_path, memory_mode, n_top_genes=10, hvg_batch_key='batch', geneset_path=geneset_path)
    pianist.initialize_features()

    # The 10 HVGs ranked among the gene-set genes only (the LOESS trend is fitted on them), in the AnnData's gene order
    expected = hvgs_within(make_adata(), GENE_SET_30, 10, 'batch')
    assert len(expected) == 10
    assert list(pianist.var_names) == expected


def test_n_top_genes_within_gene_set_trains_and_saves(tmp_path):
    # One end-to-end run: the HVGs within the gene set are trained on and saved next to the model checkpoint
    geneset_path = write_gene_set(tmp_path, GENE_SET_30)
    pianist = make_composer(make_adata(), tmp_path, n_top_genes=10, hvg_batch_key='batch', geneset_path=geneset_path)
    pianist.run_pipeline()

    expected = hvgs_within(make_adata(), GENE_SET_30, 10, 'batch')
    assert list(pianist.var_names) == expected
    assert pianist.get_latent_representation().shape == (N_CELLS, SMALL_MODEL['latent_size'])
    saved = pd.read_csv(f'{pianist.checkpoint_path}/var_names.csv', header=None, dtype=str)
    assert saved[0].tolist() == expected


@pytest.mark.parametrize('memory_mode', ALL_MODES)
@pytest.mark.parametrize('n_top_genes', ['default', None, 0, 30], ids=['default', 'None', 'zero', 'equal_to_found'])
def test_whole_gene_set_used_without_warning(n_top_genes, memory_mode, tmp_path):
    # The default (None), n_top_genes <= 0, and n_top_genes equal to the genes found all use the whole set, silently.
    # The file lists 32 genes, of which 30 are found, so 30 is equal to the genes found (not the genes listed)
    geneset_path = write_gene_set(tmp_path, GENE_SET_30 + ['not_a_gene_1', 'not_a_gene_2'])
    kwargs = {} if n_top_genes == 'default' else {'n_top_genes': n_top_genes}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        pianist = make_composer(make_data(memory_mode, tmp_path), tmp_path, memory_mode, geneset_path=geneset_path, **kwargs)
    assert n_top_genes_warnings(record) == []
    pianist.initialize_features()
    assert list(pianist.var_names) == in_adata_order(make_adata().var_names, GENE_SET_30)


@pytest.mark.parametrize('memory_mode', ALL_MODES)
@pytest.mark.parametrize('n_top_genes', [31, 50])
def test_n_top_genes_above_gene_set_size_warns_at_creation(n_top_genes, memory_mode, tmp_path):
    # Previously more genes than the set holds were asked for without any message.
    # The file lists 32 genes, of which 30 are found: n_top_genes=31 is above the genes found but not the genes listed
    geneset_path = write_gene_set(tmp_path, GENE_SET_30 + ['not_a_gene_1', 'not_a_gene_2'])
    adata = make_data(memory_mode, tmp_path)
    with pytest.warns(UserWarning) as record:
        pianist = make_composer(adata, tmp_path, memory_mode, n_top_genes=n_top_genes, geneset_path=geneset_path)
    # The warning names both numbers: the n_top_genes asked for and the 30 gene-set genes found (not the 32 listed)
    messages = n_top_genes_warnings(record)
    assert any(re.search(rf'\b{n_top_genes}\b', message) and re.search(r'\b30\b', message) for message in messages), [str(warning.message) for warning in record]
    assert not any(re.search(r'\b32\b', message) for message in messages), messages

    # All the gene-set genes are used
    pianist.initialize_features()
    assert list(pianist.var_names) == in_adata_order(make_adata().var_names, GENE_SET_30)


# ---------------------------------------------------------------------------
# The default n_top_genes without a gene set: 4096 HVGs
# ---------------------------------------------------------------------------
def make_wide_adata(memory_mode, tmp_path):
    # More genes than the default 4096 HVGs, so the default selects some of them; few cells keep it fast
    rng = np.random.default_rng(0)
    n_cells, n_genes = 100, 4200  # More cells than the batch size (64)
    mu = 10 ** rng.uniform(-1, 1, n_genes) * rng.uniform(0.5, 2.0, (n_cells, n_genes))
    counts = rng.negative_binomial(2, 2 / (2 + mu)).astype(np.float32)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], n_cells))}, index=[f'cell_{i}' for i in range(n_cells)])
    adata = ad.AnnData(X=csr_matrix(counts), obs=obs, var=pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)]))
    if memory_mode != 'backed':
        return adata
    adata.write_h5ad(tmp_path / 'wide.h5ad')  # Backed mode reads from disk; written to tmp_path
    return ad.read_h5ad(tmp_path / 'wide.h5ad', backed='r')


@pytest.mark.parametrize('memory_mode', ALL_MODES)
@pytest.mark.parametrize('n_top_genes', ['default', None], ids=['default', 'None'])
def test_default_n_top_genes_without_gene_set_is_4096_hvgs(n_top_genes, memory_mode, tmp_path):
    # Previously n_top_genes=None failed at initialize_features: '<=' not supported between 'NoneType' and 'int'
    adata = make_wide_adata(memory_mode, tmp_path)
    kwargs = {} if n_top_genes == 'default' else {'n_top_genes': n_top_genes}
    pianist = make_composer(adata, tmp_path, memory_mode, **kwargs)
    pianist.initialize_features()

    expected = highly_variable_genes(make_wide_adata('CPU', tmp_path), n_top_genes=4096, inplace=False)
    assert len(pianist.var_names) == 4096
    assert list(pianist.var_names) == list(expected.index[expected['highly_variable'].to_numpy()])


# ---------------------------------------------------------------------------
# n_top_genes is checked when the Composer is created: None or an integer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('n_top_genes', ['10', 10.5, float('nan'), True, [10]], ids=['str', 'fraction', 'nan', 'bool', 'list'])
@pytest.mark.parametrize('geneset', [False, True], ids=['no_gene_set', 'gene_set'])
def test_non_integer_n_top_genes_raises_at_creation(n_top_genes, geneset, tmp_path):
    # Previously these were accepted at creation (some failed later, True acted as 1, and with a gene set all were ignored)
    kwargs = {'geneset_path': write_gene_set(tmp_path, GENE_SET_30)} if geneset else {}
    with pytest.raises(ValueError, match='n_top_genes'):
        make_composer(make_adata(), tmp_path, n_top_genes=n_top_genes, **kwargs)


@pytest.mark.parametrize('n_top_genes', [np.int64(10), 10.0], ids=['numpy_integer', 'integer_valued_float'])
@pytest.mark.parametrize('geneset', [False, True], ids=['no_gene_set', 'gene_set'])
def test_integer_like_n_top_genes_accepted(n_top_genes, geneset, tmp_path):
    # numpy integers and floats with an integer value (e.g. 4096.0) select that many genes
    adata = make_adata()
    if geneset:
        pianist = make_composer(adata, tmp_path, n_top_genes=n_top_genes, hvg_batch_key='batch', geneset_path=write_gene_set(tmp_path, GENE_SET_30))
        expected = hvgs_within(make_adata(), GENE_SET_30, 10, 'batch')
    else:
        pianist = make_composer(adata, tmp_path, n_top_genes=n_top_genes, hvg_batch_key='batch')
        hvg = highly_variable_genes(make_adata(), n_top_genes=10, batch_key='batch', inplace=False)
        expected = list(hvg.index[hvg['highly_variable'].to_numpy()])
    pianist.initialize_features()
    assert list(pianist.var_names) == expected
