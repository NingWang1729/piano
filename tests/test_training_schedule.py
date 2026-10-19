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
from scipy.sparse import csr_matrix

from piano import Composer

CLASS_SIZES = {'x': 200, 'y': 80, 'z': 20}  # Uneven, so a cap of 50 cells per class trims x and y but not z
N_GENES = 50


# ---------------------------------------------------------------------------
# Helper: small count matrix with a batch covariate and uneven cell type classes
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    cell_types = np.repeat(list(CLASS_SIZES), list(CLASS_SIZES.values()))
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], len(cell_types))),
        'cell_type': pd.Categorical(cell_types),
    }, index=[str(i) for i in range(len(cell_types))])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(len(cell_types), N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.mark.parametrize('epoch, expected', [(0, 0.0), (100, 0.125), (200, 0.25), (300, 0.25), (399, 0.25)])
def test_annealed_weight_stops_at_its_maximum(epoch, expected):
    # Annealing over 200 epochs from 0 to 0.25; training may run longer, e.g. --max_epochs 400 with the CLI's default 200
    pianist = Composer(make_adata(), memory_mode='CPU', compile_model=False, n_top_genes=-1, deterministic=False)
    weight = pianist._get_warmup(epoch, batch_idx=0, n_batches=10, min_weight=0.0, max_weight=0.25, n_annealing_epochs=200)
    # Previously the weight kept growing after annealing (0.499 at epoch 399)
    assert weight == pytest.approx(expected)
