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
from scipy.sparse import csr_array

from piano import Composer
from piano.utils.data import to_uint16_csr

N_CELLS = 200
N_GENES = 50
LARGE_COUNTS = [65535, 65536, 70000, 1_000_000]  # The largest uint16, then counts a direct uint16 cast would wrap around


# ---------------------------------------------------------------------------
# Helper: small dense count matrix with a batch covariate and a few counts too large for uint16
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    X[0, :len(LARGE_COUNTS)] = LARGE_COUNTS
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.mark.parametrize('to_input', [np.asarray, csr_array], ids=['dense', 'csr_array'])
def test_counts_above_uint16_max_are_clipped(to_input, capsys):
    counts = np.array([[0, 3, *LARGE_COUNTS]], dtype=np.float32)
    X = to_input(counts)
    stored = to_uint16_csr(X)

    # Previously 65536, 70000 and 1e6 were stored as 0, 4464 and 16960
    assert stored.dtype == np.uint16
    assert stored.toarray().tolist() == [[0, 3, 65535, 65535, 65535, 65535]]
    assert 'Warning: 3 counts in adata.X are above 65535' in capsys.readouterr().out

    # The caller's matrix is unchanged, even when the CSR matrix shares its data
    assert np.array_equal(X.toarray() if hasattr(X, 'toarray') else X, counts)


def test_counts_within_uint16_range_are_kept_without_warning(capsys):
    counts = np.array([[0, 1, 2, 65535]], dtype=np.float32)
    assert to_uint16_csr(counts).toarray().tolist() == [[0, 1, 2, 65535]]
    assert 'Warning' not in capsys.readouterr().out


@pytest.mark.parametrize('memory_mode', [
    'SparseCPU',
    pytest.param('SparseGPU', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='SparseGPU needs a CUDA GPU')),
])
def test_sparse_modes_train_on_clipped_counts(memory_mode, tmp_path):
    # Sparse memory modes store dense counts as uint16; the batches they return hold the clipped counts
    pianist = Composer(
        make_adata(), memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(tmp_path),
    )
    pianist.run_pipeline()

    first_cell = pianist.train_adataset[torch.tensor([0])].cpu()
    assert first_cell[0, :len(LARGE_COUNTS)].tolist() == [65535] * len(LARGE_COUNTS)
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model.latent_size)
