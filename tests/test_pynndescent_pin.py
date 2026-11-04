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


import os
import tomllib

import anndata as ad
import numpy as np
import scanpy as sc
from packaging.requirements import Requirement

# Repository root, so the test reads this checkout's pyproject.toml
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# scanpy 1.11 finds neighbours exactly below 8,192 cells (Euclidean) and with pynndescent from there on
SCANPY_PYNNDESCENT_MIN_CELLS = 8_192


# ---------------------------------------------------------------------------
# pynndescent 0.7.0 returns a csr_array that scanpy 1.11 cannot read (`getnnz`), so every CPU sc.pp.neighbors on
# 8,192 cells or more failed, as in the command line's UMAPs (SCRUM-198). piano pins pynndescent below 0.7
# (fails before this change: pynndescent was not declared at all)
# ---------------------------------------------------------------------------
def test_pyproject_pins_pynndescent_below_0_7():
    with open(os.path.join(REPO_ROOT, 'pyproject.toml'), 'rb') as f:
        dependencies = [Requirement(r) for r in tomllib.load(f)['project']['dependencies']]
    pins = [r for r in dependencies if r.name == 'pynndescent']
    assert len(pins) == 1, 'pynndescent is not a declared dependency'
    assert not pins[0].specifier.contains('0.7.0'), pins[0]
    assert pins[0].specifier.contains('0.6.0'), pins[0]


def test_scanpy_neighbors_work_where_scanpy_uses_pynndescent():
    """
    The command line's CPU UMAP step on enough cells for scanpy to use pynndescent: it builds the neighbour graph and
    the UMAP (fails with pynndescent 0.7.0 installed, which the pin prevents).
    """
    n_cells, n_dims, n_neighbors = SCANPY_PYNNDESCENT_MIN_CELLS + 1_000, 10, 15
    rng = np.random.default_rng(0)
    adata = ad.AnnData(obs={'cell': np.arange(n_cells).astype(str)})
    adata.obsm['X_latent'] = rng.normal(size=(n_cells, n_dims)).astype(np.float32)

    # As piano/__main__.py: neighbours on a stored representation, then the UMAP
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_latent', random_state=0)
    sc.tl.umap(adata, random_state=0, maxiter=10)

    distances = adata.obsp['distances']
    assert distances.shape == (n_cells, n_cells)
    # Every cell has at least its n_neighbors - 1 nearest other cells (on this path scanpy 1.11 also stores more
    # entries per cell: 16 for 15 neighbours, measured)
    neighbours_per_cell = np.diff(distances.indptr)
    assert neighbours_per_cell.min() >= n_neighbors - 1, np.bincount(neighbours_per_cell)
    assert adata.obsm['X_umap'].shape == (n_cells, 2)
    assert np.isfinite(adata.obsm['X_umap']).all()
