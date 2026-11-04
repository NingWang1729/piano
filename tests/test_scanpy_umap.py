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


import ast
import os

import anndata as ad
import numpy as np
import pytest
import scanpy as sc
import scipy.sparse.csgraph
import scipy.sparse.linalg

from piano.utils import scanpy_umap as scanpy_umap_module
from piano.utils.scanpy_umap import ARPACK_MAX_ENTRIES, eigensolver, scanpy_umap

# Repository root, so the tests read this checkout's command line
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def latent_adata(n_clusters, cells_per_cluster, n_dims=10, separation=0.0, seed=0):
    """Cells in n_clusters Gaussian clusters, with their neighbour graph as the command line builds it."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=separation, size=(n_clusters, n_dims))
    latent = np.concatenate([center + rng.normal(size=(cells_per_cluster, n_dims)) for center in centers])
    adata = ad.AnnData(obs={'cell': np.arange(len(latent)).astype(str)})
    adata.obsm['X_latent'] = latent.astype(np.float32)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_latent', random_state=0)
    return adata


def n_connected_components(adata):
    return scipy.sparse.csgraph.connected_components(adata.obsp['connectivities'])[0]


# ---------------------------------------------------------------------------
# umap-learn uses ARPACK (eigsh) below 2M cells per connected component, whose n x sqrt(n) Lanczos vectors overflow its
# 32-bit indices from ~1.66M cells: a segmentation fault (SCRUM-199). piano's copy uses lobpcg there instead
# (fails before this change: piano.utils.scanpy_umap did not exist)
# ---------------------------------------------------------------------------
def first_overflowing_size(dim):
    """The fewest vertices for which eigsh's Lanczos vectors pass ARPACK's 32-bit indices."""
    n = 1
    while n * max(2 * (dim + 1) + 1, int(np.sqrt(n))) <= ARPACK_MAX_ENTRIES:
        n = n * 2 if n < 1_000_000 else n + 1_000
    # Refine to the exact size
    while (n - 1) * max(2 * (dim + 1) + 1, int(np.sqrt(n - 1))) > ARPACK_MAX_ENTRIES:
        n -= 1
    return n


@pytest.mark.parametrize('dim', [2, 3])
def test_eigensolver_uses_lobpcg_where_arpack_would_overflow(dim):
    overflow = first_overflowing_size(dim)
    assert 1_600_000 < overflow < 1_700_000
    assert eigensolver(overflow - 1, dim) == 'eigsh'
    assert eigensolver(overflow, dim) == 'lobpcg'
    # The AIT33 training set of the CodeOcean capsule: one connected component that crashed sc.tl.umap
    assert eigensolver(1_851_543, dim) == 'lobpcg'
    # umap-learn's own choices stay: eigsh for small graphs, lobpcg from 2M
    assert eigensolver(10_000, dim) == 'eigsh'
    assert eigensolver(1_500_000, dim) == 'eigsh'
    assert eigensolver(2_000_000, dim) == 'lobpcg'


def test_spectral_start_uses_lobpcg_for_components_above_the_bound(monkeypatch):
    """
    With the bound lowered to a few thousand entries, a small connected graph takes the lobpcg path: eigsh is never
    called, and the UMAP is finite.
    """
    adata = latent_adata(n_clusters=1, cells_per_cluster=1_000)
    assert n_connected_components(adata) == 1

    def no_eigsh(*args, **kwargs):
        raise AssertionError('eigsh called for a component above the ARPACK bound')

    monkeypatch.setattr(scanpy_umap_module, 'ARPACK_MAX_ENTRIES', 10_000)
    monkeypatch.setattr(scipy.sparse.linalg, 'eigsh', no_eigsh)
    scanpy_umap(adata, random_state=0, maxiter=20)
    assert np.isfinite(adata.obsm['X_umap']).all()


# ---------------------------------------------------------------------------
# Below the bound nothing changes: for the same random_state, piano's copy gives exactly sc.tl.umap's UMAP, with one
# connected component or several, and leaves the neighbour graph as it is
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('n_clusters, separation, expected_components', [
    (1, 0.0, 1),     # One connected component: the spectral embedding of the whole graph
    (3, 100.0, 3),   # A few components (at most 2 x dim): placed on the axes, then embedded one by one
    (6, 100.0, 6),   # Many components (more than 2 x dim): placed by a spectral embedding of their centroids
])
@pytest.mark.parametrize('random_state', [0, 1])
def test_scanpy_umap_equals_sc_tl_umap(n_clusters, separation, expected_components, random_state):
    adata = latent_adata(n_clusters, cells_per_cluster=300, separation=separation)
    assert n_connected_components(adata) == expected_components

    expected, result = adata.copy(), adata.copy()
    sc.tl.umap(expected, random_state=random_state)
    scanpy_umap(result, random_state=random_state)

    np.testing.assert_array_equal(result.obsm['X_umap'], expected.obsm['X_umap'])
    assert result.uns['umap'] == expected.uns['umap']
    # The graph is not pruned in place (sc.tl.umap's .tocoo() shares its values with .obsp['connectivities'])
    assert (result.obsp['connectivities'] != adata.obsp['connectivities']).nnz == 0


def test_scanpy_umap_equals_sc_tl_umap_above_10000_cells():
    """Above 10,000 cells umap-learn's default is 200 epochs (500 below); here 20 epochs, as maxiter, to keep it short."""
    adata = latent_adata(n_clusters=1, cells_per_cluster=10_500, n_dims=5)
    expected, result = adata.copy(), adata.copy()
    sc.tl.umap(expected, random_state=0, maxiter=20)
    scanpy_umap(result, random_state=0, maxiter=20)
    np.testing.assert_array_equal(result.obsm['X_umap'], expected.obsm['X_umap'])


# ---------------------------------------------------------------------------
# The command line's CPU UMAPs use piano's copy, not sc.tl.umap (rapids' UMAP on the GPU, as before)
# ---------------------------------------------------------------------------
def test_command_line_does_not_call_sc_tl_umap():
    with open(os.path.join(REPO_ROOT, 'piano', '__main__.py')) as f:
        tree = ast.parse(f.read())
    calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert 'sc.tl.umap' not in calls
    assert calls.count('compute_umap') == 4
