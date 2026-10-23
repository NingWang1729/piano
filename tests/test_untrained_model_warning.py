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

import re
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
UNTRAINED_WARNING = r'neither trained nor loaded.*random weights'


# ---------------------------------------------------------------------------
# Helpers: small count matrix with a batch covariate, and Composers whose model is prepared, trained or loaded
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


def make_composer(outdir, memory_mode='CPU'):
    """A fresh Composer. Without a CUDA GPU, GPU modes fall back to CPU modes at creation, with a warning."""
    outdir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # Only the memory_mode fallback warning is expected here
        warnings.filterwarnings('ignore', message=r"memory_mode='.*' needs a CUDA GPU")
        return Composer(
            make_data(memory_mode, outdir), memory_mode=memory_mode, compile_model=False,
            categorical_covariate_keys=['batch'], n_top_genes=-1, max_epochs=1, batch_size=64, early_stopping=False,
            deterministic=False, outdir=str(outdir),
        )


def prepared_composer(outdir, memory_mode='CPU'):
    """A Composer whose model is built (prepare_model) but has random weights: neither trained nor loaded."""
    pianist = make_composer(outdir, memory_mode)
    pianist.prepare_model()
    return pianist


def trained_composer(outdir):
    """A Composer trained for one epoch; it writes its model checkpoint to outdir / 'checkpoints'."""
    pianist = make_composer(outdir)
    pianist.run_pipeline()
    return pianist


def checkpoint_of(pianist):
    return f'{pianist.checkpoint_path}/model_checkpoint.pt'


def untrained_warnings(function, *args, **kwargs):
    """Call function and return its result and the untrained-model warnings it gave (others are ignored)."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        result = function(*args, **kwargs)
    return result, [warning for warning in record if is_untrained_warning(warning)]


def is_untrained_warning(warning):
    return re.search(UNTRAINED_WARNING, str(warning.message)) is not None


def expected_shape(method, pianist):
    return (N_CELLS, pianist.model_kwargs['latent_size']) if method == 'get_latent_representation' else (N_CELLS, N_GENES)


# ---------------------------------------------------------------------------
# A model that was built but neither trained nor loaded warns: its output comes from random weights
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('data', ['training', 'query'])
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
@pytest.mark.parametrize('method', METHODS)
def test_prepared_but_untrained_model_warns(method, memory_mode, data, tmp_path):
    pianist = prepared_composer(tmp_path, memory_mode)
    adata = None if data == 'training' else make_data(memory_mode, tmp_path, seed=1)

    # Previously returned output from random weights without any message
    result, caught = untrained_warnings(getattr(pianist, method), adata)
    # One warning per call, naming the user's call (here, in untrained_warnings in this file)
    assert [warning.category for warning in caught] == [UserWarning]
    assert [warning.filename for warning in caught] == [__file__]
    assert result.shape == expected_shape(method, pianist)


def test_composer_loaded_from_untrained_pickle_warns(tmp_path):
    # The pickle of a prepared but untrained Composer holds a model with random weights, so it warns after load too
    prepared_composer(tmp_path).save(tmp_path / 'pianist.pkl')
    pianist = Composer.load(tmp_path / 'pianist.pkl')
    for method in METHODS:
        _, caught = untrained_warnings(getattr(pianist, method), make_adata(seed=1))
        assert len(caught) == 1, method


def test_model_prepared_again_after_training_warns(tmp_path):
    # prepare_model builds a new model with random weights, replacing the trained one
    pianist = trained_composer(tmp_path)
    pianist.prepare_model()
    for method in METHODS:
        _, caught = untrained_warnings(getattr(pianist, method), make_adata(seed=1))
        assert len(caught) == 1, method


def test_composer_without_model_raises_without_untrained_warning(tmp_path):
    # The model check comes first: a Composer with no model gets the ValueError only
    pianist = make_composer(tmp_path)
    for method in METHODS:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            with pytest.raises(ValueError, match='no model'):
                getattr(pianist, method)()
        assert not [warning for warning in record if is_untrained_warning(warning)], method


# ---------------------------------------------------------------------------
# Coverage: trained or loaded models do not warn
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('method', METHODS)
def test_trained_model_does_not_warn(method, tmp_path):
    pianist = trained_composer(tmp_path)
    for adata in (None, make_adata(seed=1)):
        result, caught = untrained_warnings(getattr(pianist, method), adata)
        assert caught == []
        assert result.shape == expected_shape(method, pianist)


@pytest.mark.parametrize('method', METHODS)
def test_model_with_weights_from_load_model_does_not_warn(method, tmp_path):
    # A Composer that was never trained itself, given trained weights with load_model
    checkpoint = checkpoint_of(trained_composer(tmp_path / 'trained'))
    pianist = prepared_composer(tmp_path / 'loaded')
    pianist.load_model(checkpoint)

    result, caught = untrained_warnings(getattr(pianist, method), make_adata(seed=1))
    assert caught == []
    assert result.shape == expected_shape(method, pianist)


@pytest.mark.parametrize('method', METHODS)
def test_composer_loaded_from_trained_pickle_does_not_warn(method, tmp_path):
    # Without a checkpoint, Composer.load keeps the pickled trained model
    trained_composer(tmp_path).save(tmp_path / 'pianist.pkl')
    pianist = Composer.load(tmp_path / 'pianist.pkl')

    result, caught = untrained_warnings(getattr(pianist, method), make_adata(seed=1))
    assert caught == []
    assert result.shape == expected_shape(method, pianist)


@pytest.mark.parametrize('method', METHODS)
def test_untrained_pickle_loaded_with_checkpoint_does_not_warn(method, tmp_path):
    # Composer.load with a checkpoint loads trained weights (through load_model) into a Composer pickled before training
    checkpoint = checkpoint_of(trained_composer(tmp_path / 'trained'))
    prepared_composer(tmp_path / 'prepared').save(tmp_path / 'pianist.pkl')
    pianist = Composer.load(tmp_path / 'pianist.pkl', model_checkpoint_path=checkpoint)

    result, caught = untrained_warnings(getattr(pianist, method), make_adata(seed=1))
    assert caught == []
    assert result.shape == expected_shape(method, pianist)
