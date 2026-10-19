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
from piano.utils.data import to_stored_csr

N_CELLS = 200
N_GENES = 50
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
MEMORY_MODES = ['CPU', 'SparseCPU', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)]

# Counts stored as two entries for the same cell and gene: (cell, gene, first entry, second entry, their sum as stored).
# 40000 + 40000 is clipped to 65535 as a whole; 0.5 + 0.5 is the integer 1
DUPLICATES = [(0, 0, 40000, 40000, 65535), (0, 1, 0.5, 0.5, 1), (1, 0, 300, 200, 500)]


# ---------------------------------------------------------------------------
# Helpers: an unsummed CSR matrix (duplicate entries) and the uint16 counts it should give
# ---------------------------------------------------------------------------
def make_counts(seed=0):
    """Poisson counts with the DUPLICATES' cells and genes zeroed, to be filled by the duplicate entries."""
    X = np.random.default_rng(seed).poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    for cell, gene, *_ in DUPLICATES:
        X[cell, gene] = 0
    return X


def make_unsummed_csr(counts, to_format=csr_matrix):
    """counts as CSR, plus each of the DUPLICATES as two entries placed after the row's other entries."""
    rows, cols = np.nonzero(counts)
    values = counts[rows, cols]
    for cell, gene, first, second, _ in DUPLICATES:
        rows = np.append(rows, [cell, cell])
        cols = np.append(cols, [gene, gene])
        values = np.append(values, np.float32([first, second]))
    order = np.argsort(rows, kind='stable')  # Group entries by row, keeping the duplicates last in their rows
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=counts.shape[0]))])
    X = to_format((values[order], cols[order], indptr), shape=counts.shape)
    assert not X.has_canonical_format
    return X


def expected_counts(counts):
    """The counts every memory mode should store: each duplicated count summed, then clipped."""
    expected = counts.copy()
    for cell, gene, *_, stored in DUPLICATES:
        expected[cell, gene] = stored
    return expected


def assert_conversion_messages(out):
    """The 3 duplicate entries are summed with a note; only the summed 80000 is clipped; the summed 0.5s are integers."""
    assert 'Note: adata.X has 3 duplicate entries' in out
    assert 'Warning: 1 counts in adata.X are above 65535' in out
    assert 'not integers' not in out


@pytest.mark.parametrize('to_format', [csr_matrix, csr_array], ids=lambda f: f.__name__)
def test_duplicate_entries_are_summed_before_conversion(to_format, capsys):
    # Previously each entry was rounded and clipped on its own and then summed: 40000 + 40000 wrapped to 14464 with no
    # clipping warning, and 0.5 + 0.5 gave 0
    counts = make_counts()
    X = make_unsummed_csr(counts, to_format)
    original = (X.data.copy(), X.indices.copy(), X.indptr.copy())

    stored = to_stored_csr(X)
    assert stored.dtype == np.uint16
    assert np.array_equal(stored.toarray(), expected_counts(counts))
    assert_conversion_messages(capsys.readouterr().out)

    # The caller's matrix is unchanged, although csr_matrix(X) shares its arrays and sum_duplicates works in place
    for array, original_array in zip((X.data, X.indices, X.indptr), original):
        assert np.array_equal(array, original_array)


def test_canonical_csr_is_used_without_note(capsys):
    # A CSR without duplicate entries (the usual case) prints no note
    counts = make_counts()
    stored = to_stored_csr(csr_matrix(counts))
    assert np.array_equal(stored.toarray(), counts)
    assert 'duplicate' not in capsys.readouterr().out


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_memory_modes_sum_duplicate_entries(memory_mode, tmp_path, capsys):
    # Every in-memory mode gives the summed counts. Previously SparseGPU's Triton kernel also kept only one of the
    # duplicates (300 + 200 gave 300), unlike SparseCPU (500)
    counts = make_counts()
    X = make_unsummed_csr(counts)
    original = (X.data.copy(), X.indices.copy(), X.indptr.copy())
    obs = pd.DataFrame({'batch': pd.Categorical(np.tile(['a', 'b'], N_CELLS // 2))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]

    pianist = Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(tmp_path),
    )
    pianist.initialize_features()
    pianist.prepare_data()
    batch = pianist.train_adataset[torch.arange(N_CELLS)].cpu()
    assert torch.equal(batch[:, :N_GENES], torch.from_numpy(expected_counts(counts)))
    assert_conversion_messages(capsys.readouterr().out)

    # adata.X is unchanged, duplicates included
    for array, original_array in zip((adata.X.data, adata.X.indices, adata.X.indptr), original):
        assert np.array_equal(array, original_array)
