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
import scanpy as sc
from scipy.sparse import csr_matrix

from piano import highly_variable_genes

N_CELLS = 900
N_GENES = 300
N_TOP_GENES = 60
BATCHES = ['a', 'b', 'c']
STAT_COLUMNS = ['means', 'variances', 'variances_norm', 'highly_variable_rank', 'highly_variable_nbatches']


# ---------------------------------------------------------------------------
# Helper: negative binomial counts with gene means spanning several orders of magnitude,
# batch-specific scaling, genes expressed in only one batch, and one all-zero gene
# ---------------------------------------------------------------------------
def make_adata(sparse=True, seed=0):
    rng = np.random.default_rng(seed)
    batch = rng.choice(BATCHES, N_CELLS)
    gene_means = 10 ** rng.uniform(-2, 1.5, N_GENES)
    batch_scale = {b: rng.uniform(0.5, 2.0, N_GENES) for b in BATCHES}
    mu = gene_means * np.stack([batch_scale[b] for b in batch])
    dispersion = 2.0
    counts = rng.negative_binomial(dispersion, dispersion / (dispersion + mu)).astype(np.float32)
    counts[batch != 'a', :5] = 0  # Expressed only in batch a (zero variance in b and c)
    counts[:, 5] = 0  # Not expressed anywhere
    obs = pd.DataFrame({'batch': pd.Categorical(batch)}, index=[f'cell_{i}' for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(counts) if sparse else counts, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def scanpy_hvgs(adata, batch_key):
    return sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=N_TOP_GENES,
                                       batch_key=batch_key, inplace=False).set_index(adata.var_names)


def assert_matches_scanpy(piano_df, scanpy_df, batch_key):
    assert piano_df['highly_variable'].sum() == N_TOP_GENES
    np.testing.assert_array_equal(piano_df['highly_variable'].to_numpy(), scanpy_df['highly_variable'].to_numpy())
    columns = STAT_COLUMNS if batch_key is not None else [c for c in STAT_COLUMNS if c != 'highly_variable_nbatches']
    for column in columns:
        np.testing.assert_allclose(piano_df[column].to_numpy(dtype=np.float64), scanpy_df[column].to_numpy(dtype=np.float64),
                                   rtol=1e-6, atol=1e-10, equal_nan=True, err_msg=column)


@pytest.mark.parametrize('sparse', [True, False], ids=['sparse', 'dense'])
@pytest.mark.parametrize('batch_key', ['batch', None], ids=['batch_key', 'no_batch_key'])
def test_one_adata_matches_scanpy(sparse, batch_key):
    adata = make_adata(sparse=sparse)
    piano_df = highly_variable_genes(adata, n_top_genes=N_TOP_GENES, batch_key=batch_key, inplace=False)
    assert_matches_scanpy(piano_df, scanpy_hvgs(adata, batch_key), batch_key)


@pytest.mark.parametrize('split', ['by_batch', 'mixed'])
@pytest.mark.parametrize('batch_key', ['batch', None], ids=['batch_key', 'no_batch_key'])
def test_multiple_adatas_match_scanpy_on_concatenation(split, batch_key):
    # Split one AnnData into parts: one part per batch, or uneven parts that each hold cells from several batches
    adata = make_adata()
    if split == 'by_batch':
        parts = [adata[adata.obs['batch'] == b].copy() for b in BATCHES]
    else:
        parts = [adata[start:stop].copy() for start, stop in [(0, 100), (100, 550), (550, N_CELLS)]]
    piano_df = highly_variable_genes(parts, n_top_genes=N_TOP_GENES, batch_key=batch_key, inplace=False)
    assert_matches_scanpy(piano_df, scanpy_hvgs(ad.concat(parts), batch_key), batch_key)


def test_inplace_writes_same_hvgs_to_every_adata():
    parts = [make_adata(seed=1)[:400].copy(), make_adata(seed=1)[400:].copy()]
    expected = highly_variable_genes(parts, n_top_genes=N_TOP_GENES, batch_key='batch', inplace=False)
    assert highly_variable_genes(parts, n_top_genes=N_TOP_GENES, batch_key='batch', inplace=True) is None
    for part in parts:
        assert part.uns['hvg'] == {'flavor': 'seurat_v3'}
        np.testing.assert_array_equal(part.var['highly_variable'].to_numpy(), expected['highly_variable'].to_numpy())
        for column in STAT_COLUMNS:
            np.testing.assert_array_equal(part.var[column].to_numpy(), expected[column].to_numpy())


@pytest.mark.parametrize('backed', [False, True], ids=['in_memory', 'backed'])
@pytest.mark.parametrize('second_genes', ['reordered', 'subset'])
def test_adatas_with_different_var_names_raise(second_genes, backed, tmp_path):
    # Statistics are combined by gene position: previously reordered genes silently gave wrong means and HVGs
    adata = make_adata()
    parts = [adata[:400].copy(), adata[400:, ::-1].copy() if second_genes == 'reordered' else adata[400:, :-1].copy()]
    if backed:
        for index, part in enumerate(parts):
            part.write_h5ad(tmp_path / f'part_{index}.h5ad')
        parts = [ad.read_h5ad(tmp_path / f'part_{index}.h5ad', backed='r') for index in range(len(parts))]
    with pytest.raises(ValueError, match='adata 1 has different var_names'):
        highly_variable_genes(parts, n_top_genes=N_TOP_GENES, inplace=False)


@pytest.mark.parametrize('dtype', ['category', 'object'])
def test_missing_batch_labels_raise(dtype):
    # Previously failed while sorting the labels with "TypeError: '<' not supported between instances of 'float' and 'str'"
    parts = [make_adata()[:400].copy(), make_adata()[400:].copy()]
    parts[1].obs['batch'] = parts[1].obs['batch'].astype(dtype)
    parts[1].obs.loc[parts[1].obs_names[:3], 'batch'] = np.nan
    with pytest.raises(ValueError, match=r"adata 1\.obs\['batch'\] has 3 missing \(NaN\) batch labels"):
        highly_variable_genes(parts, n_top_genes=N_TOP_GENES, batch_key='batch', inplace=False)
