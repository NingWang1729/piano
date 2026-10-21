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
from piano.utils.data import to_stored_csr, to_stored_dense

N_CELLS = 200
N_GENES = 50
LARGE_COUNTS = [65535, 65536, 70000, 1_000_000]  # The largest uint16, then counts a direct uint16 cast would wrap around
NON_INTEGERS = [0.4, 2.5, 2.7, 3.5, 65535.6]
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
MEMORY_MODES = ['CPU', 'SparseCPU', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)]


# ---------------------------------------------------------------------------
# Helper: small count matrix with a batch covariate and a few counts too large for uint16, optionally non-integer
# ---------------------------------------------------------------------------
def make_adata(to_format=np.asarray, non_integer=False, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    X[0, :len(LARGE_COUNTS)] = LARGE_COUNTS
    if non_integer:
        X[1, :len(NON_INTEGERS)] = NON_INTEGERS
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=to_format(X), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(adata, memory_mode, outdir):
    return Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir),
    )


@pytest.mark.parametrize('to_input', [np.asarray, csr_array], ids=['dense', 'csr_array'])
def test_integer_counts_are_stored_as_uint16_and_clipped(to_input, capsys):
    counts = np.array([[0, 3, *LARGE_COUNTS]], dtype=np.float32)
    X = to_input(counts)
    stored = to_stored_csr(X)

    # Counts above 65535 are clipped; a direct cast would store 65536, 70000 and 1e6 as 0, 4464 and 16960
    assert stored.dtype == np.uint16
    assert stored.toarray().tolist() == [[0, 3, 65535, 65535, 65535, 65535]]
    out = capsys.readouterr().out
    assert 'Warning: 3 counts in adata.X are above 65535' in out
    assert 'float32' not in out

    # The caller's matrix is unchanged, even when the CSR matrix shares its data
    assert np.array_equal(X.toarray() if hasattr(X, 'toarray') else X, counts)


@pytest.mark.parametrize('to_input', [np.asarray, csr_array], ids=['dense', 'csr_array'])
def test_non_integer_counts_are_stored_as_float32_unrounded(to_input, capsys):
    counts = np.array([[0, 3, *NON_INTEGERS, 70000]], dtype=np.float32)
    X = to_input(counts)
    stored = to_stored_csr(X)

    # Previously these were rounded (2.7 to 3, 0.4 to 0) and 70000 clipped, as uint16; now they are kept as given
    assert stored.dtype == np.float32
    assert np.array_equal(stored.toarray(), counts)
    out = capsys.readouterr().out
    assert 'Note: 5 values in adata.X are not integers, so counts are stored as float32' in out
    assert 'clipping' not in out

    # The caller's matrix is unchanged
    assert np.array_equal(X.toarray() if hasattr(X, 'toarray') else X, counts)


@pytest.mark.parametrize('chunk_size', [1, 2, 8192])
@pytest.mark.parametrize('non_integer', [False, True], ids=['integers', 'non_integers'])
def test_dense_storage_matches_csr_storage(non_integer, chunk_size, capsys):
    # Dense counts are read in row chunks; dtype, values and messages match the CSR conversion for any chunk size
    counts = np.array([[0, 2, 3, 1], [4, 65535, 70000, 1], [5, 0, 1e6, 2]], dtype=np.float32)
    if non_integer:
        counts[2, 3] = 2.7  # In the last chunk when chunk_size is 1
    expected = to_stored_csr(counts).toarray()
    expected_messages = capsys.readouterr().out

    stored = to_stored_dense(counts, chunk_size=chunk_size)
    assert stored.dtype == expected.dtype == (np.float32 if non_integer else np.uint16)
    assert np.array_equal(stored, expected)
    assert capsys.readouterr().out == expected_messages
    assert counts[1, 2] == 70000  # The caller's counts are unchanged


def test_counts_within_uint16_range_are_kept_without_messages(capsys):
    counts = np.array([[0, 1, 2, 65535]], dtype=np.float32)
    stored = to_stored_csr(counts)
    assert stored.dtype == np.uint16 and stored.toarray().tolist() == [[0, 1, 2, 65535]]
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('memory_mode', [*MEMORY_MODES, 'backed'])
@pytest.mark.parametrize('to_format', [np.asarray, csr_matrix], ids=['dense', 'csr_matrix'])
def test_memory_modes_train_on_clipped_integer_counts(to_format, memory_mode, tmp_path):
    # Every in-memory mode stores integer counts of any format as uint16, clipped to 65535; backed mode clips them as it
    # reads them from disk (previously it kept them), so every mode trains and embeds on the same counts
    adata = make_adata(to_format)
    if memory_mode == 'backed':
        # Backed mode reads a .h5ad file, written to tmp_path
        adata.write_h5ad(tmp_path / 'counts.h5ad')
        adata = ad.read_h5ad(tmp_path / 'counts.h5ad', backed='r')
    pianist = make_composer(adata, memory_mode, tmp_path)
    pianist.run_pipeline()

    first_cell = pianist.train_adataset[torch.tensor([0])].cpu()
    assert first_cell[0, :len(LARGE_COUNTS)].tolist() == [65535] * len(LARGE_COUNTS)
    caller_counts = ad.read_h5ad(tmp_path / 'counts.h5ad').X if memory_mode == 'backed' else adata.X
    assert caller_counts[0, 2] == 70000  # The caller's counts (and file) are unchanged
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model.latent_size)


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
@pytest.mark.parametrize('to_format', [np.asarray, csr_matrix], ids=['dense', 'csr_matrix'])
def test_memory_modes_train_on_non_integer_counts_as_given(to_format, memory_mode, tmp_path, capsys):
    # With any non-integer count, every in-memory mode stores float32 counts unrounded and unclipped, and says so.
    # Previously 2.7 was stored as 3 and 70000 as 65535
    adata = make_adata(to_format, non_integer=True)
    pianist = make_composer(adata, memory_mode, tmp_path)
    pianist.run_pipeline()

    first_cells = pianist.train_adataset[torch.tensor([0, 1])].cpu()
    assert first_cells[0, :len(LARGE_COUNTS)].tolist() == LARGE_COUNTS
    assert torch.equal(first_cells[1, :len(NON_INTEGERS)], torch.tensor(NON_INTEGERS, dtype=torch.float32))
    assert 'Note: 5 values in adata.X are not integers' in capsys.readouterr().out
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model.latent_size)


@pytest.mark.parametrize('memory_mode', ['CPU', 'SparseCPU', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)])
def test_counterfactuals_are_re_encoded_as_they_are(memory_mode, tmp_path):
    # The CLI embeds counterfactual counts (float NB means) with get_latent_representation; they are no longer rounded
    pianist = make_composer(make_adata(), memory_mode, tmp_path)
    pianist.run_pipeline()
    adata_cf = make_adata()
    adata_cf.X = pianist.get_counterfactual()

    batch = pianist._get_adataset(adata_cf, memory_mode)[torch.arange(N_CELLS)].cpu()
    assert torch.equal(batch[:, :N_GENES], torch.from_numpy(adata_cf.X.astype(np.float32)))
    assert pianist.get_latent_representation(adata_cf).shape == (N_CELLS, pianist.model.latent_size)


@pytest.mark.parametrize('memory_mode', ['CPU', pytest.param('GPU', marks=GPU)])
@pytest.mark.parametrize('non_integer', [False, True], ids=['integers', 'non_integers'])
def test_dense_modes_store_counts_exactly(memory_mode, non_integer, tmp_path):
    # Dense modes return the counts as given (as float32), with the batch covariate after them: 2 bytes per integer count
    # (including 32768 and 65535, beyond the int16 range), 4 bytes per count when any is not an integer
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    X[0, :2] = [32768, 65535]
    if non_integer:
        X[1, 0] = 0.5
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    pianist = make_composer(ad.AnnData(X=X, obs=obs), memory_mode, tmp_path)
    pianist.initialize_features()
    pianist.prepare_data()

    batch = pianist.train_adataset[torch.arange(N_CELLS)].cpu()
    assert batch.dtype == torch.float32
    assert torch.equal(batch[:, :N_GENES], torch.from_numpy(X))
    assert torch.equal(batch[:, N_GENES:].sum(dim=1), torch.ones(N_CELLS))  # One-hot batch covariate
    assert pianist.train_adataset.counts.element_size() == (4 if non_integer else 2)
