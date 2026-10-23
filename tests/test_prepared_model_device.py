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

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')

MEMORY_MODES = ['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed']


# ---------------------------------------------------------------------------
# Helpers: a Composer with a prepared but untrained model, on a small count matrix with a batch covariate
# ---------------------------------------------------------------------------
def make_adata():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def prepare_composer(tmp_path, memory_mode='CPU'):
    """A Composer after initialize_features, prepare_data and prepare_model, without training."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    adata = make_adata()
    if memory_mode == 'backed':
        # Backed mode reads the counts from disk, so the AnnData is written to tmp_path first
        adata.write_h5ad(tmp_path / 'adata.h5ad')
        adata = ad.read_h5ad(tmp_path / 'adata.h5ad', backed='r')
    pianist = Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        batch_size=64, early_stopping=False, deterministic=False, outdir=str(tmp_path),
    )
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model()
    return pianist


def parameter_device_types(model):
    """The device types of the model's parameters, e.g. {'cuda'}."""
    return {parameter.device.type for parameter in model.parameters()}


@pytest.fixture
def no_gpu(monkeypatch):
    """Pretend that no CUDA GPU is available, so the Composer's device is 'cpu' on GPU nodes too."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


# ---------------------------------------------------------------------------
# prepare_model puts the model on the Composer's device, so inference works before training
# ---------------------------------------------------------------------------
@GPU
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_prepared_model_is_on_the_composers_device(memory_mode, tmp_path):
    # With a GPU, the Composer's device is 'cuda' in every memory mode (the CPU modes keep only the data on the CPU)
    pianist = prepare_composer(tmp_path, memory_mode)
    assert pianist.device == 'cuda'
    assert parameter_device_types(pianist.model) == {'cuda'}
    assert pianist.model.device == pianist.device
    # Moving the model does not switch it to eval mode: a freshly prepared model stays in training mode
    assert pianist.model.training


@GPU
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_inference_on_prepared_model(memory_mode, tmp_path):
    # Previously the model stayed on the CPU until train(), while batches went to the GPU, so inference failed with
    # "Expected all tensors to be on the same device"
    pianist = prepare_composer(tmp_path, memory_mode)
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model_kwargs['latent_size'])
    assert pianist.get_counterfactual().shape == (N_CELLS, N_GENES)


def test_prepared_model_stays_on_cpu_without_gpu(no_gpu, tmp_path):
    # Coverage: a CPU Composer keeps its model on the CPU, and inference before training works
    pianist = prepare_composer(tmp_path)
    assert pianist.device == 'cpu'
    assert parameter_device_types(pianist.model) == {'cpu'}
    assert pianist.model.device == 'cpu'
    assert pianist.model.training
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model_kwargs['latent_size'])


@GPU
def test_initial_weights_do_not_depend_on_the_device(monkeypatch, tmp_path):
    # Coverage: the weights are built on the CPU and moved afterwards, so a GPU Composer starts from the same weights as a
    # CPU Composer with the same random seed
    gpu_weights = prepare_composer(tmp_path / 'gpu').model.state_dict()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    cpu_weights = prepare_composer(tmp_path / 'cpu').model.state_dict()
    assert gpu_weights.keys() == cpu_weights.keys()
    for key, cpu_value in cpu_weights.items():
        assert torch.equal(gpu_weights[key].cpu(), cpu_value), key
