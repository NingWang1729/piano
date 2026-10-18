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
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')  # Render without a display

from piano.utils import umap as piano_umap
from piano.utils.umap import generate_distinct_colors, plot_umaps


# ---------------------------------------------------------------------------
# Helper: AnnData with a random 2D embedding and a label with n categories
# ---------------------------------------------------------------------------
def make_labeled_adata(n_categories, cells_per_category=3, seed=0):
    n_cells = n_categories * cells_per_category
    labels = pd.Categorical([f'type_{i % n_categories}' for i in range(n_cells)])
    adata = ad.AnnData(obs=pd.DataFrame({'label': labels}, index=[str(i) for i in range(n_cells)]))
    adata.obsm['X_umap'] = np.random.default_rng(seed).normal(size=(n_cells, 2))
    return adata


@pytest.mark.parametrize('n', [1, 20, 338])
def test_generate_distinct_colors(n):
    colors = generate_distinct_colors(n)
    assert len(colors) == n
    assert len(set(colors)) == n  # No repeated colors
    assert all(matplotlib.colors.is_color_like(c) for c in colors)


# ---------------------------------------------------------------------------
# Scanpy's largest default palette has 102 colors, so 338 and 600 categories
# need the custom palette; legends are dropped at >= 500 categories.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('n_categories, expect_legend', [(20, True), (338, True), (600, False)])
def test_plot_umaps_colors_every_category(n_categories, expect_legend, tmp_path, monkeypatch):
    # plot_umaps closes each figure after saving; keep them open so they can be inspected
    figures = []
    monkeypatch.setattr(piano_umap.plt, 'close', figures.append)

    # Plot one label and check the figure was saved
    plot_umaps(make_labeled_adata(n_categories), ['label'], outdir=str(tmp_path))
    assert (tmp_path / 'X_umap__label.png').exists()
    ax = figures[0].axes[0]

    # One distinct color per category means no category fell back to gray
    face_colors = np.unique(ax.collections[0].get_facecolors(), axis=0)
    assert len(face_colors) == n_categories

    # Legend is kept below 500 categories and removed at or above
    assert (ax.get_legend() is not None) == expect_legend
