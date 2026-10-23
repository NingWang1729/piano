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

from piano.utils.data import AnnDataset, ConcatAnnDataset

N_CELLS = (6, 5)  # Two datasets, so global index 6 is the first cell of the second dataset
N_GENES = 7
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
# SparseCPU, SparseGPU and backed are left out: their single datasets have their own scalar-index bugs (separate issue)
MEMORY_MODES = ['CPU', pytest.param('GPU', marks=GPU)]

# Each scalar type makes the same integer index from a Python int, given the device the concat dataset is on
INDEX_TYPES = {
    'int': lambda i, device: i,
    'np.int64': lambda i, device: np.int64(i),
    'np.int32': lambda i, device: np.int32(i),
    '0-d_cpu_tensor': lambda i, device: torch.tensor(i),
    '0-d_device_tensor': lambda i, device: torch.tensor(i, device=device),
}


# ---------------------------------------------------------------------------
# Helpers: two small in-memory datasets with a batch covariate, concatenated
# ---------------------------------------------------------------------------
def make_adata(n_cells, seed):
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n_cells, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], n_cells), categories=['a', 'b'])},
                       index=[f'cell_{seed}_{i}' for i in range(n_cells)])
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_concat(memory_mode):
    """ConcatAnnDataset of two datasets, and the counts of all its cells in order."""
    adatas = [make_adata(n_cells, seed) for seed, n_cells in enumerate(N_CELLS)]
    concat = ConcatAnnDataset([AnnDataset(adata, memory_mode=memory_mode, categorical_covariate_keys=['batch']) for adata in adatas])
    counts = np.vstack([adata.X for adata in adatas])
    return concat, counts


# ---------------------------------------------------------------------------
# A scalar index of any integer type gives the same 1-D row as a Python int
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
@pytest.mark.parametrize('index_type', list(INDEX_TYPES.values()), ids=list(INDEX_TYPES))
@pytest.mark.parametrize('i', [2, 6, 10], ids=['first_dataset', 'second_dataset_start', 'last_cell'])
def test_integer_scalar_index_returns_one_row(memory_mode, index_type, i):
    # Before the fix, numpy and 0-d tensor scalars raised "IndexError: too many indices for tensor of dimension 0"
    concat, counts = make_concat(memory_mode)
    row = concat[index_type(i, concat.device)]

    assert row.shape == concat[i].shape == (N_GENES + 2,)  # Counts, then the one-hot batch covariate
    assert torch.equal(row, concat[i])
    assert torch.equal(row, concat[[i]][0])
    assert np.array_equal(row[:N_GENES].cpu().numpy(), counts[i])


# ---------------------------------------------------------------------------
# Batch indices still give a 2-D batch, in the given order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
@pytest.mark.parametrize('make_batch', [list, np.array], ids=['list', 'numpy_array'])
def test_batch_index_returns_rows_in_order(memory_mode, make_batch):
    concat, counts = make_concat(memory_mode)
    indices = [7, 1, 10, 3]  # Crosses the boundary between the datasets both ways
    batch = concat[make_batch(indices)]

    assert batch.shape == (len(indices), N_GENES + 2)
    assert np.array_equal(batch[:, :N_GENES].cpu().numpy(), counts[indices])


# Bools are not row numbers: Python True used to give row 1 silently (bool is a subclass of int), and a 0-d bool tensor
# took the integer path too; every kind of bool scalar now raises
@pytest.mark.parametrize('flag', [True, np.bool_(True), torch.tensor(True)], ids=['python_bool', 'numpy_bool', '0-d_bool_tensor'])
def test_bool_scalar_index_raises(flag):
    concat, _ = make_concat('CPU')
    with pytest.raises(TypeError, match='not a bool'):
        concat[flag]

