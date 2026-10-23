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

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from piano import Composer

N_CELLS = 200
N_GENES = 50
MEMORY_MODES = ['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed']
METHODS = ['get_latent_representation', 'get_counterfactual']
NO_MODEL_ERROR = r'no model.*run_pipeline.*Composer\.load'


# ---------------------------------------------------------------------------
# Helpers: small count matrix with a batch covariate, and a Composer that was never trained
#
# Every file these tests write (backed .h5ad files, pickles, checkpoints) goes to tmp_path, the new, empty temporary
# directory pytest gives each test, so nothing is written to the repository or next to any data.
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_data(memory_mode, tmp_path, seed=0):
    """make_adata(seed), read back from an .h5ad in tmp_path with backed='r' in backed mode."""
    adata = make_adata(seed)
    if memory_mode != 'backed':
        return adata
    path = tmp_path / f'adata_{seed}.h5ad'
    adata.write_h5ad(path)
    return ad.read_h5ad(path, backed='r')


def make_composer(memory_mode, tmp_path):
    """A fresh Composer (no model). Without a CUDA GPU, GPU modes fall back to CPU modes at creation, with a warning."""
    with warnings.catch_warnings():
        # Only the memory_mode fallback warning is expected here; the model check comes before the memory mode is used
        warnings.filterwarnings('ignore', message=r"memory_mode='.*' needs a CUDA GPU")
        return Composer(
            make_data(memory_mode, tmp_path), memory_mode=memory_mode, compile_model=False,
            categorical_covariate_keys=['batch'], n_top_genes=-1, max_epochs=1, batch_size=64, early_stopping=False,
            deterministic=False, outdir=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# A Composer without a model raises a clear ValueError first, before building features or datasets
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('data', ['training', 'query'])
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
@pytest.mark.parametrize('method', METHODS)
def test_untrained_composer_raises(method, memory_mode, data, tmp_path):
    pianist = make_composer(memory_mode, tmp_path)
    adata = None if data == 'training' else make_data(memory_mode, tmp_path, seed=1)

    # Previously initialized the features, then failed with "AttributeError: 'NoneType' object has no attribute ..."
    with pytest.raises(ValueError, match=NO_MODEL_ERROR):
        getattr(pianist, method)(adata)
    # The check comes first, so the failed call did no work and left the Composer unchanged
    assert not pianist.initialized_features
    assert pianist.model is None


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
@pytest.mark.parametrize('method', METHODS)
def test_composer_saved_before_training_raises_after_load(method, memory_mode, tmp_path):
    # Only with a query: adata=None after Composer.load already raises its own error (SCRUM-30)
    make_composer(memory_mode, tmp_path).save(tmp_path / 'pianist.pkl')
    pianist = Composer.load(tmp_path / 'pianist.pkl')
    assert pianist.model is None

    # Previously failed with "TypeError: 'NoneType' object is not subscriptable" in initialize_features
    with pytest.raises(ValueError, match=NO_MODEL_ERROR):
        getattr(pianist, method)(make_data(memory_mode, tmp_path, seed=1))
    assert not pianist.initialized_features


def test_counterfactual_model_check_before_argument_checks(tmp_path):
    # Without a model nothing can succeed, so the error that says how to get one comes before the covariates error
    pianist = make_composer('CPU', tmp_path)
    with pytest.raises(ValueError, match=NO_MODEL_ERROR):
        pianist.get_counterfactual(covariates='bogus')
