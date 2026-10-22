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
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

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


def train_composer(outdir):
    """A Composer trained for one epoch on make_adata(), on the GPU when one is available."""
    pianist = Composer(
        make_adata(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir),
    )
    pianist.run_pipeline()  # Writes the model checkpoint to outdir / 'checkpoints'
    return pianist


@pytest.fixture
def trained_composer(tmp_path):
    return train_composer(tmp_path)


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


# ---------------------------------------------------------------------------
# Loading on other hardware: after Composer.load, the Composer's device, the model's device and the memory mode match it
#
# A Composer trained on a GPU may be loaded on a machine without one. Tests marked GPU need a CUDA GPU. Most others pretend
# that none is available (no_gpu), so that they also run, and fail on old code, on GPU nodes.
# ---------------------------------------------------------------------------
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')


@pytest.fixture
def no_gpu(monkeypatch):
    """Pretend that no CUDA GPU is available."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


@pytest.fixture
def trained_composer_without_gpu(no_gpu, tmp_path):
    """A Composer trained on the CPU (so its pickle has no CUDA tensors), with no CUDA GPU available from here on."""
    return train_composer(tmp_path)


def checkpoint_of(pianist):
    return f'{pianist.checkpoint_path}/model_checkpoint.pt'


def parameter_devices(model):
    """The devices of the model's parameters, e.g. {'cpu'}."""
    return {str(parameter.device) for parameter in model.parameters()}


def device_warnings(record):
    """The recorded warnings about the device ("device='cuda' ..."), out of all recorded warnings."""
    return [warning for warning in record if "device='" in str(warning.message)]


# Without a checkpoint, load moves the pickled model to the device and sets eval mode, as load_model does with a checkpoint
def test_load_without_checkpoint_sets_eval_mode(trained_composer, tmp_path):
    # This guards consistency with load_model, which sets eval mode: get_latent_representation and get_counterfactual
    # switch to eval mode and back, so model.training is the only visible effect. Previously the model was left in
    # training mode, as it was pickled after training
    trained_composer.save(tmp_path / 'pianist.pkl')
    loaded = Composer.load(tmp_path / 'pianist.pkl', device='cpu')

    assert not loaded.model.training


@GPU
def test_load_without_checkpoint_moves_gpu_model_to_device(trained_composer, tmp_path):
    assert all(parameter.is_cuda for parameter in trained_composer.model.parameters())
    trained_composer.save(tmp_path / 'pianist.pkl')

    # Previously only the Composer's device was set: the parameters stayed on cuda:0, model.device 'cuda'
    loaded = Composer.load(tmp_path / 'pianist.pkl', device='cpu')

    assert loaded.device == 'cpu'
    assert loaded.model.device == 'cpu'
    assert parameter_devices(loaded.model) == {'cpu'}
    assert not loaded.model.training


# load and load_model default to the available device, and fall back to the CPU, with a warning, without a GPU
@pytest.mark.parametrize('with_checkpoint', [True, False], ids=['with_checkpoint', 'without_checkpoint'])
@pytest.mark.parametrize('device', ['cuda', 'cuda:0', torch.device('cuda')], ids=['cuda', 'cuda:0', 'torch.device'])
def test_load_to_cuda_without_gpu_falls_back_to_cpu(trained_composer_without_gpu, tmp_path, device, with_checkpoint):
    pianist = trained_composer_without_gpu
    pianist.save(tmp_path / 'pianist.pkl')
    model_checkpoint_path = checkpoint_of(pianist) if with_checkpoint else None

    # Previously did not warn: without a checkpoint, the Composer's device was set to the CUDA device anyway, and with one,
    # torch.load(map_location='cuda') failed
    with pytest.warns(UserWarning, match=r"device='cuda(:0)?' needs a CUDA GPU.*device='cpu'") as record:
        loaded = Composer.load(tmp_path / 'pianist.pkl', model_checkpoint_path=model_checkpoint_path, device=device)
    # One warning, at the user's Composer.load(...) line
    assert [warning.filename for warning in device_warnings(record)] == [__file__]

    assert loaded.device == 'cpu'
    assert loaded.model.device == 'cpu'
    assert parameter_devices(loaded.model) == {'cpu'}


# Coverage: the default device was already 'cpu'
@pytest.mark.parametrize('with_checkpoint', [True, False], ids=['with_checkpoint', 'without_checkpoint'])
def test_load_defaults_to_cpu_without_gpu(trained_composer_without_gpu, tmp_path, with_checkpoint):
    pianist = trained_composer_without_gpu
    pianist.save(tmp_path / 'pianist.pkl')
    model_checkpoint_path = checkpoint_of(pianist) if with_checkpoint else None

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        loaded = Composer.load(tmp_path / 'pianist.pkl', model_checkpoint_path=model_checkpoint_path)
    assert device_warnings(record) == []

    assert loaded.device == 'cpu'
    assert loaded.model.device == 'cpu'
    assert parameter_devices(loaded.model) == {'cpu'}


@GPU
@pytest.mark.parametrize('with_checkpoint', [True, False], ids=['with_checkpoint', 'without_checkpoint'])
def test_load_defaults_to_gpu(trained_composer, tmp_path, with_checkpoint):
    trained_composer.save(tmp_path / 'pianist.pkl')
    model_checkpoint_path = checkpoint_of(trained_composer) if with_checkpoint else None

    # Previously the default device was 'cpu' on GPU nodes too. No warning about moving the model: with a checkpoint, the
    # model is rebuilt on the CPU before load_model moves it to 'cuda', but the Composer is already on 'cuda', and the
    # warning compares with the Composer's device, not with where the parameters are
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        loaded = Composer.load(tmp_path / 'pianist.pkl', model_checkpoint_path=model_checkpoint_path)
    assert device_warnings(record) == []

    assert loaded.device == 'cuda'
    assert loaded.model.device == 'cuda'
    assert all(parameter.is_cuda for parameter in loaded.model.parameters())


def test_load_model_to_cuda_without_gpu_falls_back_to_cpu(trained_composer_without_gpu):
    pianist = trained_composer_without_gpu

    # Previously did not warn, and torch.load(map_location='cuda') failed
    with pytest.warns(UserWarning, match=r"device='cuda' needs a CUDA GPU.*device='cpu'") as record:
        pianist.load_model(checkpoint_of(pianist), device='cuda')
    # One warning, at the user's load_model(...) line: the Composer stays on the CPU, so the model does not move
    assert [warning.filename for warning in device_warnings(record)] == [__file__]

    assert pianist.device == 'cpu'
    assert pianist.model.device == 'cpu'
    assert parameter_devices(pianist.model) == {'cpu'}


def test_load_model_to_another_device_warns(trained_composer_without_gpu):
    pianist = trained_composer_without_gpu
    pianist.device = 'cuda'  # As for a Composer created or loaded on a GPU node

    # Previously moved the model to the CPU without a warning
    with pytest.warns(UserWarning, match=r"model from device='cuda' to device='cpu'") as record:
        pianist.load_model(checkpoint_of(pianist), device='cpu')
    # One warning, at the user's load_model(...) line
    assert [warning.filename for warning in device_warnings(record)] == [__file__]

    assert pianist.device == 'cpu'
    assert pianist.model.device == 'cpu'


def test_load_model_without_gpu_on_a_cuda_composer_warns_once(trained_composer_without_gpu):
    pianist = trained_composer_without_gpu
    pianist.device = 'cuda'  # As for a GPU Composer read with a plain pickle.load, which keeps its device

    # Previously did not warn, and loaded to the CPU. Only the fallback warning: device is empty, so the warning about
    # moving the model, which tells the user to leave device empty, would mislead
    with pytest.warns(UserWarning, match=r"device='cuda' needs a CUDA GPU.*device='cpu'") as record:
        pianist.load_model(checkpoint_of(pianist))
    assert [warning.filename for warning in device_warnings(record)] == [__file__]

    assert pianist.device == 'cpu'
    assert pianist.model.device == 'cpu'
    assert parameter_devices(pianist.model) == {'cpu'}


def test_load_model_defaults_to_composer_device(trained_composer):
    device = trained_composer.device

    # Previously the default device was 'cpu', so on a GPU node the model moved off the GPU (coverage on CPU-only nodes)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        trained_composer.load_model(checkpoint_of(trained_composer))
    assert device_warnings(record) == []

    assert trained_composer.device == device
    assert {parameter.device.type for parameter in trained_composer.model.parameters()} == {torch.device(device).type}


# Coverage: a bare 'cuda' is the current CUDA device, so these do not move the model of a Composer on 'cuda'
@GPU
@pytest.mark.parametrize(
    'device', ['cuda', 'cuda:0', torch.device('cuda'), torch.device('cuda', 0)],
    ids=['cuda', 'cuda:0', 'torch.device', 'torch.device_0'],
)
def test_load_model_to_the_same_gpu_does_not_warn(trained_composer, device):
    assert trained_composer.device == 'cuda'

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        trained_composer.load_model(checkpoint_of(trained_composer), device=device)
    assert device_warnings(record) == []

    assert all(parameter.is_cuda for parameter in trained_composer.model.parameters())
