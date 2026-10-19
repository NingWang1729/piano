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
from scipy.sparse import csr_array, csr_matrix

from piano import Composer
from piano.utils.data import to_uint16_csr, to_uint16_dense

N_CELLS = 200
N_GENES = 50
LARGE_COUNTS = [65535, 65536, 70000, 1_000_000]  # The largest uint16, then counts a direct uint16 cast would wrap around


# ---------------------------------------------------------------------------
# Helper: small count matrix with a batch covariate, a few counts too large for uint16 and one non-integer count
# ---------------------------------------------------------------------------
def make_adata(to_format=np.asarray, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    X[0, :len(LARGE_COUNTS)] = LARGE_COUNTS
    X[1, 0] = 2.7
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=to_format(X), obs=obs)
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


@pytest.mark.parametrize('to_input', [np.asarray, csr_array], ids=['dense', 'csr_array'])
def test_non_integer_counts_are_rounded(to_input, capsys):
    counts = np.array([[0, 0.4, 2.5, 2.7, 3.5, 65535.6]], dtype=np.float32)
    X = to_input(counts)
    stored = to_uint16_csr(X)

    # Previously the cast truncated (2.7 to 2, 3.5 to 3); 65535.6 rounds to 65536, which is then clipped
    assert stored.toarray().tolist() == [[0, 0, 2, 3, 4, 65535]]
    out = capsys.readouterr().out
    assert 'Warning: 5 counts in adata.X are not integers' in out
    assert 'Warning: 1 counts in adata.X are above 65535' in out

    # The caller's matrix is unchanged, even when the CSR matrix shares its data
    assert np.array_equal(X.toarray() if hasattr(X, 'toarray') else X, counts)


@pytest.mark.parametrize('chunk_size', [1, 2, 8192])
def test_dense_conversion_matches_csr_conversion(chunk_size, capsys):
    # Dense counts are converted in row chunks; values and warning totals match the CSR conversion for any chunk size
    counts = np.array([[0, 0.4, 2.5, 2.7], [3.5, 65535.6, 70000, 1], [5, 0, 1e6, 2]], dtype=np.float32)
    expected = to_uint16_csr(counts).toarray()
    expected_warnings = capsys.readouterr().out

    stored = to_uint16_dense(counts, chunk_size=chunk_size)
    assert stored.dtype == np.uint16
    assert np.array_equal(stored, expected)
    assert capsys.readouterr().out == expected_warnings
    assert 'Warning: 5 counts in adata.X are not integers' in expected_warnings
    assert 'Warning: 3 counts in adata.X are above 65535' in expected_warnings
    assert counts[1, 2] == 70000  # The caller's counts are unchanged


def test_counts_within_uint16_range_are_kept_without_warning(capsys):
    counts = np.array([[0, 1, 2, 65535]], dtype=np.float32)
    assert to_uint16_csr(counts).toarray().tolist() == [[0, 1, 2, 65535]]
    assert 'Warning' not in capsys.readouterr().out


GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
MEMORY_MODES = ['CPU', 'SparseCPU', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)]


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
@pytest.mark.parametrize('to_format', [np.asarray, csr_matrix], ids=['dense', 'csr_matrix'])
def test_memory_modes_train_on_uint16_counts(to_format, memory_mode, tmp_path):
    # Every in-memory mode stores counts of any format as uint16; the batches it returns hold the rounded and clipped counts.
    # Previously GPU and CPU modes kept counts as float32 (70000 and 2.7 stayed as they were), unlike sparse modes
    adata = make_adata(to_format)
    pianist = Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(tmp_path),
    )
    pianist.run_pipeline()

    first_cells = pianist.train_adataset[torch.tensor([0, 1])].cpu()
    assert first_cells[0, :len(LARGE_COUNTS)].tolist() == [65535] * len(LARGE_COUNTS)
    assert first_cells[1, 0] == 3
    assert adata.X[0, 2] == 70000  # The caller's counts are unchanged
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model.latent_size)


@pytest.mark.parametrize('memory_mode', ['CPU', pytest.param('GPU', marks=GPU)])
def test_dense_modes_store_integer_counts_exactly_in_two_bytes(memory_mode, tmp_path):
    # Raw integer counts come back unchanged (as float32), with the batch covariate after them, from 2 bytes per count
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    X[0, :2] = [32768, 65535]  # Counts above the largest int16 survive the int16 storage
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    pianist = Composer(
        ad.AnnData(X=X, obs=obs), memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'],
        n_top_genes=-1, max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(tmp_path),
    )
    pianist.initialize_features()
    pianist.prepare_data()

    batch = pianist.train_adataset[torch.arange(N_CELLS)].cpu()
    assert batch.dtype == torch.float32
    assert torch.equal(batch[:, :N_GENES], torch.from_numpy(X))
    assert torch.equal(batch[:, N_GENES:].sum(dim=1), torch.ones(N_CELLS))  # One-hot batch covariate
    assert pianist.train_adataset.counts.element_size() == 2
