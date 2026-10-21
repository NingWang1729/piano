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
        pianist = make_composer(adata, tmp_path, memory_mode, n_top_genes=N_TOP_GENES, geneset_path=geneset_path)
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
        make_composer(adata, tmp_path, memory_mode, n_top_genes=N_TOP_GENES, geneset_path=geneset_path)


@pytest.mark.parametrize('memory_mode', ALL_MODES)
def test_empty_gene_set_raises_at_creation(memory_mode, tmp_path):
    # Previously the file was only read when genes were selected, failing there with pandas' EmptyDataError
    geneset_path = write_gene_set(tmp_path, [])
    adata = make_data(memory_mode, tmp_path)
    with pytest.raises(ValueError, match=re.escape(geneset_path)):
        make_composer(adata, tmp_path, memory_mode, n_top_genes=N_TOP_GENES, geneset_path=geneset_path)


# ---------------------------------------------------------------------------
# Training AnnDatas that already have exactly the selected genes share their counts (copy_adata=False, the default)
# ---------------------------------------------------------------------------
# Backed mode reads the counts from disk and never copies them: test_backed_adata_shares_nothing checks it does not warn
SHARED_COUNTS_WARNING = 'copy_adata=True'  # The sharing warning says how to get a full copy


def add_other_fields(adata):
    # Fields the Composer never reads, which its own AnnData drops (the counts may be large, and so may these)
    adata.layers['counts'] = adata.X.copy()
    adata.obsm['X_pca'] = np.zeros((adata.n_obs, 2), dtype=np.float32)
    adata.obsp['distances'] = csr_matrix((adata.n_obs, adata.n_obs), dtype=np.float32)
    adata.varm['PCs'] = np.zeros((adata.n_vars, 2), dtype=np.float32)
    adata.uns['note'] = 'not used by the Composer'
    return adata


def shares_counts(adata, other):
    # True if the two AnnDatas' count matrices share memory (the test data are CSR, so compare their values)
    return np.shares_memory(adata.X.data, other.X.data)


def shared_counts_warnings(record):
    return [str(warning.message) for warning in record if SHARED_COUNTS_WARNING in str(warning.message)]


def create_and_initialize(adata, tmp_path, memory_mode, **kwargs):
    # Return the Composer after initialize_features, with the sharing warnings raised at creation and at initialize_features
    with warnings.catch_warnings(record=True) as at_creation:
        warnings.simplefilter('always')
        pianist = make_composer(adata, tmp_path, memory_mode, **kwargs)
    with warnings.catch_warnings(record=True) as at_initialization:
        warnings.simplefilter('always')
        pianist.initialize_features()
    return pianist, shared_counts_warnings(at_creation), shared_counts_warnings(at_initialization)


def assert_only_counts_and_annotations(adata):
    # The Composer's own AnnData keeps only X, obs and var
    assert len(adata.layers) == 0
    assert len(adata.obsm) == 0
    assert len(adata.obsp) == 0
    assert len(adata.varm) == 0
    assert len(adata.uns) == 0


@pytest.mark.parametrize('memory_mode', IN_MEMORY_MODES)
def test_adata_with_the_selected_genes_shares_its_counts(memory_mode, tmp_path):
    # Previously the counts were copied, an extra copy of all training counts for the Composer's lifetime
    adata = add_other_fields(make_adata())
    original = adata.copy()
    pianist, at_creation, at_initialization = create_and_initialize(adata, tmp_path, memory_mode, n_top_genes=-1)

    # One warning, at creation, that the counts are shared, not to edit adata.X in place, and how to get a full copy
    assert len(at_creation) == 1, at_creation
    assert 'adata.X' in at_creation[0]
    assert at_initialization == []

    # The counts are shared; obs and var are the Composer's own, and the other fields are dropped
    assert shares_counts(pianist.adata[0], adata)
    assert_only_counts_and_annotations(pianist.adata[0])
    adata.obs['batch'] = pd.Categorical(['a'] * N_CELLS)
    adata.var['note'] = 'added by the caller'
    assert pianist.adata[0].obs['batch'].equals(original.obs['batch'])
    assert 'note' not in pianist.adata[0].var

    # Training leaves the caller's counts and other fields unchanged
    pianist.prepare_data()
    pianist.prepare_model(**pianist.model_kwargs)
    pianist.train()
    assert (adata.X != original.X).nnz == 0
    assert (adata.layers['counts'] != original.layers['counts']).nnz == 0
    assert adata.uns['note'] == original.uns['note']
    assert pianist.get_latent_representation().shape == (N_CELLS, SMALL_MODEL['latent_size'])


@pytest.mark.parametrize('memory_mode', IN_MEMORY_MODES)
def test_only_adatas_with_the_selected_genes_in_order_share_their_counts(memory_mode, tmp_path):
    # All genes of the first AnnData are selected (n_top_genes=-1); the second has them in reverse order, so it is copied
    first, second, third = make_adata(0), make_adata(1), make_adata(2)
    second = second[:, second.var_names[::-1]].copy()
    pianist, at_creation, at_initialization = create_and_initialize([first, second, third], tmp_path, memory_mode, n_top_genes=-1)

    # Previously every AnnData was copied; the warning is raised once however many AnnDatas share their counts
    assert [shares_counts(mine, theirs) for mine, theirs in zip(pianist.adata, [first, second, third])] == [True, False, True]
    assert len(at_creation) == 1, at_creation
    assert at_initialization == []

    # The copied AnnData is in the first AnnData's gene order, like the shared ones
    assert all(list(adata.var_names) == list(first.var_names) for adata in pianist.adata)
    np.testing.assert_array_equal(pianist.adata[1].X.toarray(), second[:, first.var_names].X.toarray())


@pytest.mark.parametrize('memory_mode', IN_MEMORY_MODES)
def test_copy_adata_shares_nothing(memory_mode, tmp_path):
    # copy_adata=True asks for a full copy, as all training AnnDatas were copied before (previously not an argument)
    adata = add_other_fields(make_adata())
    pianist, at_creation, at_initialization = create_and_initialize(adata, tmp_path, memory_mode, n_top_genes=-1, copy_adata=True)

    assert at_creation == [] and at_initialization == []
    assert not shares_counts(pianist.adata[0], adata)
    assert_only_counts_and_annotations(pianist.adata[0])
    np.testing.assert_array_equal(pianist.adata[0].X.toarray(), adata.X.toarray())

    # An in-place edit to the caller's counts does not reach the Composer
    adata.X.data[:] = 0
    assert pianist.adata[0].X.sum() > 0


@pytest.mark.parametrize('memory_mode', IN_MEMORY_MODES)
def test_view_with_the_selected_genes_is_copied(memory_mode, tmp_path):
    # Coverage (unchanged behaviour): a view re-slices its parent on every read, so it is still copied
    parent = make_adata()
    view = parent[:N_CELLS // 2]
    pianist, at_creation, at_initialization = create_and_initialize(view, tmp_path, memory_mode, n_top_genes=-1)

    assert at_creation == [] and at_initialization == []
    assert not pianist.adata[0].is_view
    assert not shares_counts(pianist.adata[0], parent)  # A weak check: slicing the rows of a CSR matrix already copies its values
    np.testing.assert_array_equal(pianist.adata[0].X.toarray(), view.X.toarray())


@pytest.mark.parametrize('memory_mode', IN_MEMORY_MODES)
@pytest.mark.parametrize('selection', ['hvgs', 'gene_set'])
def test_adata_with_more_than_the_selected_genes_is_copied(memory_mode, selection, tmp_path):
    # Gene selection subsets the counts, so they are copied and nothing is shared; the other fields are dropped
    # (previously they were copied along: layers, obsm, obsp and uns, and varm for the selected genes)
    adata = add_other_fields(make_adata())
    gene_selection = {'n_top_genes': N_TOP_GENES}
    if selection == 'gene_set':
        gene_selection['geneset_path'] = write_gene_set(tmp_path, ['gene_3', 'gene_7', 'gene_15'])
    pianist, at_creation, at_initialization = create_and_initialize(adata, tmp_path, memory_mode, **gene_selection)

    assert at_creation == [] and at_initialization == []
    assert pianist.adata[0].n_vars < N_GENES
    assert not shares_counts(pianist.adata[0], adata)
    assert_only_counts_and_annotations(pianist.adata[0])


def test_backed_adata_shares_nothing(tmp_path):
    # Coverage (unchanged behaviour): a backed AnnData with all its genes selected meets the other conditions for sharing,
    # but backed mode reads the counts from disk, so there is nothing to share and no warning
    adata = make_data('backed', tmp_path)
    pianist, at_creation, at_initialization = create_and_initialize(adata, tmp_path, 'backed', n_top_genes=-1)

    assert at_creation == [] and at_initialization == []
    assert pianist.adata[0] is adata


@pytest.mark.parametrize('memory_mode', ['CPU', 'SparseCPU'])
def test_shared_counts_give_the_same_latents_as_a_copy(memory_mode, tmp_path):
    # Sharing the counts changes no numbers: the same seed gives the same latents as a full copy. CPU only, where
    # training with deterministic=False is still reproducible (the MIG regression check compares with master)
    latents = []
    for copy_adata in (False, True):
        pianist = make_composer(make_adata(), tmp_path / str(copy_adata), memory_mode, n_top_genes=-1, copy_adata=copy_adata, random_seed=0)
        pianist.run_pipeline()
        latents.append(pianist.get_latent_representation())
    np.testing.assert_array_equal(latents[0], latents[1])


def test_editing_callers_obs_after_training_leaves_training_data_unchanged(tmp_path):
    # Coverage guard: SparseCPU datasets read sparse continuous covariates from obs on every batch, so the Composer
    # keeps its own obs even when the counts are shared (keeping the caller's AnnData would change this counterfactual)
    rng = np.random.default_rng(0)
    adata = make_adata()
    adata.obs['drug'] = rng.choice(['A', 'B'], N_CELLS)
    adata.obs['dose'] = rng.uniform(size=N_CELLS)
    pianist = make_composer(adata, tmp_path, 'SparseCPU', n_top_genes=-1, sparse_continuous_covariate_keys=[('drug', 'dose')])
    pianist.run_pipeline()

    before = pianist.get_counterfactual()  # Expected counts (mc_samples=0), so the same for the same inputs
    adata.obs['dose'] *= 100
    np.testing.assert_array_equal(pianist.get_counterfactual(), before)
