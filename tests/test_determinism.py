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

from piano import Composer

N_CELLS = 200
N_GENES = 50


# ---------------------------------------------------------------------------
# Helpers: small count matrix with a batch covariate, and a fixture that restores torch's process-wide settings
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(tmp_path, deterministic, memory_mode='CPU'):
    return Composer(
        make_adata(), memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=deterministic, outdir=str(tmp_path),
    )


@pytest.fixture
def restore_determinism():
    # These tests turn deterministic algorithms on, which is process-wide; restore the settings for later tests
    algorithms, cudnn = torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic
    yield
    torch.use_deterministic_algorithms(algorithms)
    torch.backends.cudnn.deterministic = cudnn


def test_deterministic_false_turns_off_deterministic_algorithms(restore_determinism, tmp_path):
    make_composer(tmp_path, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic

    # Previously a later Composer with deterministic=False kept the settings an earlier Composer turned on
    make_composer(tmp_path, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_gpu_training_after_a_deterministic_composer(restore_determinism, tmp_path):
    make_composer(tmp_path, deterministic=True)

    # Previously failed in the first GPU matrix multiplication with the CuBLAS determinism error
    # ("... set CUBLAS_WORKSPACE_CONFIG ..."), as deterministic algorithms were still on
    pianist = make_composer(tmp_path, deterministic=False, memory_mode='GPU')
    pianist.run_pipeline()
    assert pianist.trained_model
