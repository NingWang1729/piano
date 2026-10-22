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
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50
NUM_WORKERS = 2

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')


# ---------------------------------------------------------------------------
# Helpers: small CSR count matrix with a batch covariate
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS)),
    }, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(X), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(outdir, memory_mode='CPU', adata=None, num_workers=NUM_WORKERS):
    return Composer(
        make_adata() if adata is None else adata, memory_mode=memory_mode, num_workers=num_workers,
        categorical_covariate_keys=['batch'], max_epochs=1, compile_model=False, n_top_genes=-1,
        batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir),
    )


@pytest.fixture
def no_gpu(monkeypatch):
    """Pretend that no CUDA GPU is available, so the fallback is tested (and fails on old code) on GPU nodes too."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


def get_inference(pianist, function, adata, memory_mode=None):
    """Call get_latent_representation or get_counterfactual on adata (training data if None) and check the result's shape."""
    if function == 'latent':
        result = pianist.get_latent_representation(adata, memory_mode=memory_mode)
        assert result.shape == (N_CELLS, pianist.model_kwargs['latent_size'])
    else:
        result = pianist.get_counterfactual(adata, memory_mode=memory_mode)
        assert result.shape == (N_CELLS, N_GENES)
    return result


# ---------------------------------------------------------------------------
# Composer creation: num_workers > 0 is refused in GPU memory modes
# ---------------------------------------------------------------------------
@GPU
@pytest.mark.parametrize('memory_mode, canonical', [('GPU', 'GPU'), ('SparseGPU', 'SparseGPU'), ('sparsegpu', 'SparseGPU')])
def test_gpu_memory_mode_with_workers_raises(memory_mode, canonical, tmp_path):
    # Previously accepted: GPU training reset num_workers to 0 with a print, and SparseGPU training crashed with
    # "Cannot re-initialize CUDA in forked subprocess". The error names the canonical memory mode
    with pytest.raises(ValueError, match=f"num_workers.*memory_mode='{canonical}'"):
        make_composer(tmp_path, memory_mode)


@GPU
def test_sparse_gpu_without_workers_trains_and_embeds(tmp_path):
    # Control: num_workers=0 is the supported way to use a GPU memory mode
    pianist = make_composer(tmp_path, 'SparseGPU', num_workers=0)
    pianist.run_pipeline()
    assert pianist.trained_model
    get_inference(pianist, 'latent', None)


@pytest.mark.parametrize('memory_mode', ['CPU', 'SparseCPU', 'backed'])
def test_cpu_memory_modes_keep_workers(memory_mode, tmp_path):
    # Coverage: worker processes work in CPU memory modes, so num_workers is kept for training
    adata = make_adata()
    if memory_mode == 'backed':
        # Backed mode reads the counts from an .h5ad file, written to tmp_path
        adata.write_h5ad(tmp_path / 'adata.h5ad')
        adata = ad.read_h5ad(tmp_path / 'adata.h5ad', backed='r')

    pianist = make_composer(tmp_path, memory_mode, adata=adata)
    pianist.run_pipeline()
    assert pianist.trained_model
    assert pianist.num_workers == NUM_WORKERS


@pytest.mark.parametrize('memory_mode', ['GPU', 'SparseGPU'])
def test_gpu_memory_mode_fallback_keeps_workers(memory_mode, no_gpu, tmp_path):
    # Coverage: without a GPU, the memory mode falls back to a CPU memory mode before num_workers is checked,
    # so there is no error and the workers are kept
    with pytest.warns(UserWarning, match=f"memory_mode='{memory_mode}'"):
        pianist = make_composer(tmp_path, memory_mode)
    assert pianist.memory_mode in ('CPU', 'SparseCPU')
    assert pianist.num_workers == NUM_WORKERS

    pianist.run_pipeline()
    assert pianist.trained_model
    assert pianist.num_workers == NUM_WORKERS
