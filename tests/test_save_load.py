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

import copy
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from piano import Composer

N_CELLS = 200
N_GENES = 50


# ---------------------------------------------------------------------------
# Helpers: small count matrix with a batch covariate, and a Composer trained on it for one epoch
#
# Every file these tests write goes to tmp_path, a pytest built-in fixture: for each test, pytest creates a new, empty
# temporary directory (a pathlib.Path under the system temp folder, e.g. /tmp/pytest-of-<user>/pytest-<N>/<test name>0)
# and passes it to any fixture or test that names a tmp_path argument. Within one test, trained_composer and the test
# get the same directory, so the Composer's checkpoints (outdir) and the pickle the test saves sit side by side, and
# different tests never see each other's files. Nothing is written to the repository or next to the data; pytest
# keeps only the directories of its last 3 runs and deletes older ones.
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.fixture
def trained_composer(tmp_path):
    pianist = Composer(
        make_adata(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(tmp_path),
    )
    pianist.run_pipeline()  # Writes the model checkpoint to tmp_path / 'checkpoints'
    return pianist


def test_save_leaves_live_composer_unchanged(trained_composer, tmp_path):
    pianist = trained_composer
    init_adata = pianist._init_params['adata']
    state_before = dict(pianist.__dict__)
    assert init_adata is not None

    pianist.save(tmp_path / 'pianist.pkl')  # Pickles the Composer to a file in this test's temporary directory

    # Previously save() set _init_params['adata'] to None on the live Composer
    assert pianist._init_params['adata'] is init_adata
    assert pianist.__dict__.keys() == state_before.keys()
    for key, value in state_before.items():
        assert pianist.__dict__[key] is value, key

    # The live Composer still works after saving
    assert pianist.get_latent_representation().shape == (N_CELLS, pianist.model.latent_size)


def test_saved_pickle_drops_the_anndata(trained_composer, tmp_path):
    trained_composer.save(tmp_path / 'pianist.pkl')
    with open(tmp_path / 'pianist.pkl', 'rb') as f:
        saved = pickle.load(f)

    # The pickle keeps the 'adata' key of the configuration, set to None, and none of the training data
    assert 'adata' in saved._init_params
    assert saved._init_params['adata'] is None
    assert saved.adata is None and saved.train_adataset is None and saved.train_adata_loader is None


def test_save_load_round_trip(trained_composer, tmp_path):
    pianist = trained_composer
    latent = pianist.get_latent_representation()
    pianist.save(tmp_path / 'pianist.pkl')

    loaded = Composer.load(tmp_path / 'pianist.pkl', model_checkpoint_path=f'{pianist.checkpoint_path}/model_checkpoint.pt')

    # The pickle drops the training data but keeps the configuration
    assert loaded.adata is None
    assert loaded._init_params.keys() == pianist._init_params.keys()

    # The loaded model embeds the data like the live one (latents are posterior means, so no seeding is needed)
    loaded_latent = loaded.get_latent_representation(make_adata())
    assert np.allclose(loaded_latent, latent, atol=1e-5)


# ---------------------------------------------------------------------------
# A loaded Composer has no training data, so latents and counterfactuals need adata
#
# The pickle drops the training AnnData (see __getstate__), so after Composer.load, adata=None has no cells to use. It
# should raise a ValueError asking for adata, in every memory mode, rather than crash with "TypeError: 'NoneType' object is
# not iterable". The check comes before any data is moved to the GPU, so the GPU memory modes run on a CPU-only node too.
# ---------------------------------------------------------------------------
def load_composer(pianist, tmp_path, with_checkpoint):
    """Save pianist to this test's tmp_path and load it back, with or without the model checkpoint."""
    pianist.save(tmp_path / 'pianist.pkl')
    model_checkpoint_path = f'{pianist.checkpoint_path}/model_checkpoint.pt' if with_checkpoint else None
    return Composer.load(tmp_path / 'pianist.pkl', model_checkpoint_path=model_checkpoint_path)


@pytest.mark.parametrize('with_checkpoint', [True, False], ids=['with_checkpoint', 'without_checkpoint'])
@pytest.mark.parametrize('memory_mode', ['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'])
@pytest.mark.parametrize('method', ['get_latent_representation', 'get_counterfactual'])
def test_loaded_composer_without_adata_raises(trained_composer, tmp_path, method, memory_mode, with_checkpoint):
    # Loading without adata is fine: it can be passed later
    loaded = load_composer(trained_composer, tmp_path, with_checkpoint)

    with pytest.raises(ValueError, match='no training AnnData.*pass adata'):
        getattr(loaded, method)(memory_mode=memory_mode)


# A copy of a Composer drops the training AnnDatas as pickling does (both use __getstate__), so adata=None raises the same
# error; previously it crashed with "TypeError: 'NoneType' object is not iterable"
@pytest.mark.parametrize('method', ['get_latent_representation', 'get_counterfactual'])
def test_copied_composer_without_adata_raises(trained_composer, method):
    copied = copy.deepcopy(trained_composer)

    with pytest.raises(ValueError, match='no training AnnData.*pass adata'):
        getattr(copied, method)()


# Coverage: passing adata after load keeps working
def test_loaded_composer_with_adata_returns_latents_and_counterfactuals(trained_composer, tmp_path):
    loaded = load_composer(trained_composer, tmp_path, with_checkpoint=True)

    assert loaded.get_latent_representation(make_adata()).shape == (N_CELLS, loaded.model.latent_size)
    assert loaded.get_counterfactual(make_adata()).shape == (N_CELLS, N_GENES)
