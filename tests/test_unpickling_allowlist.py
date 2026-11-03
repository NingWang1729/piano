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

import builtins
import collections
import inspect
import os
import pickle
import re
import struct
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import torch.storage
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')

# The text every refusal of Composer.load ends with
TRUST_MESSAGE = 'Only load files you trust'


# ---------------------------------------------------------------------------
# Composer.load refuses pickles that hold anything a Composer saved by Composer.save does not hold
#
# Each crafted pickle below would run a harmless command if it were unpickled: it creates the file 'marker' in tmp_path,
# a new, empty directory pytest makes for each test. The tests check that Composer.load raises pickle.UnpicklingError and
# that the marker does not exist, so the command did not run. On master the commands run.
# ---------------------------------------------------------------------------
class Reduce:
    """An object that pickles as a call of func with args, as a crafted pickle would hold."""
    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def __reduce__(self):
        return self.func, self.args


def write_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path


def global_name(func):
    """The module and name a pickle stores for func, e.g. 'posix.system' for os.system."""
    return f'{func.__module__}.{func.__qualname__}'


def touch_command(marker):
    return f'touch {marker}'


@pytest.mark.parametrize('func, make_args', [
    (os.system, lambda marker: (touch_command(marker),)),
    (builtins.eval, lambda marker: (f'open({str(marker)!r}, "w").close()',)),
    (subprocess.Popen, lambda marker: (['touch', str(marker)],)),
], ids=['os.system', 'builtins.eval', 'subprocess.Popen'])
def test_load_refuses_global_outside_allowlist(tmp_path, func, make_args):
    marker = tmp_path / 'marker'
    path = write_pickle(Reduce(func, *make_args(marker)), tmp_path / 'pianist.pkl')

    # Previously the command ran, and load then failed on the returned object (AttributeError)
    with pytest.raises(pickle.UnpicklingError, match=f'(?s){re.escape(global_name(func))}.*{TRUST_MESSAGE}'):
        Composer.load(path, device='cpu')
    assert not marker.exists()


def test_load_refuses_foreign_class(tmp_path):
    # A harmless class that a Composer pickle does not hold
    path = write_pickle(collections.Counter({'a': 1}), tmp_path / 'pianist.pkl')

    # Previously the Counter was unpickled, and load then failed with AttributeError
    with pytest.raises(pickle.UnpicklingError, match=f'(?s)collections\\.Counter.*{TRUST_MESSAGE}'):
        Composer.load(path, device='cpu')


# A pickle built opcode by opcode (protocol 4), which calls only allowed globals: getattr, which a Composer pickle uses
# for its model's bound methods, and Composer. It evaluates
#   getattr(getattr(getattr(getattr(Composer, '__init__'), '__globals__'), 'get')('os'), 'system')(command)
# that is, it reaches os.system through the module globals of Composer.__init__, without naming os in the pickle
def pickled_global(module, name):
    return pickled_string(module) + pickled_string(name) + pickle.STACK_GLOBAL


def pickled_string(text):
    data = text.encode()
    return pickle.BINUNICODE + struct.pack('<I', len(data)) + data


def getattr_escape_pickle(command):
    getattr_global = pickled_global('builtins', 'getattr')
    return (
        pickle.PROTO + bytes([4])
        # Four getattr callables on the stack, used from the innermost call outwards
        + getattr_global * 4
        # getattr(Composer, '__init__')
        + pickled_global('piano.utils.composer', 'Composer') + pickled_string('__init__') + pickle.TUPLE2 + pickle.REDUCE
        # getattr(..., '__globals__'), then getattr(..., 'get')
        + pickled_string('__globals__') + pickle.TUPLE2 + pickle.REDUCE
        + pickled_string('get') + pickle.TUPLE2 + pickle.REDUCE
        # ...('os'), the os module
        + pickled_string('os') + pickle.TUPLE1 + pickle.REDUCE
        # getattr(os, 'system')(command)
        + pickled_string('system') + pickle.TUPLE2 + pickle.REDUCE
        + pickled_string(command) + pickle.TUPLE1 + pickle.REDUCE
        + pickle.STOP
    )


def test_load_refuses_getattr_escape(tmp_path):
    marker = tmp_path / 'marker'
    path = tmp_path / 'pianist.pkl'
    path.write_bytes(getattr_escape_pickle(touch_command(marker)))

    # Previously the command ran (load then failed on its return value): an allowlist with plain getattr allows this
    with pytest.raises(pickle.UnpicklingError, match=r"(?s)getattr.*'__init__'"):
        Composer.load(path, device='cpu')
    assert not marker.exists()


def test_load_refuses_code_in_pickled_tensor_bytes(tmp_path):
    # torch.storage._load_from_bytes, which a Composer pickle uses for its tensors, unpickles the bytes it is given with
    # torch.load. Here the bytes are a pickle that runs the command
    marker = tmp_path / 'marker'
    payload = pickle.dumps(Reduce(os.system, touch_command(marker)))
    path = write_pickle(Reduce(torch.storage._load_from_bytes, payload), tmp_path / 'pianist.pkl')

    # Previously the command ran under torch.load(weights_only=False), which then failed with RuntimeError
    # 'Invalid magic number; corrupt file?'. piano's message replaces torch's, which suggests weights_only=False
    with pytest.raises(pickle.UnpicklingError, match=TRUST_MESSAGE) as excinfo:
        Composer.load(path, device='cpu')
    assert not marker.exists()
    # torch's own error is kept as the cause ('from err'), and its advice to set weights_only=False is not shown
    assert isinstance(excinfo.value.__cause__, pickle.UnpicklingError)
    assert 'weights_only=False' not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Coverage (passes on master): Composers saved by Composer.save still load, in every memory mode and model configuration
#
# The pickle, the model checkpoint and the query data are written to tmp_path.
# ---------------------------------------------------------------------------
def make_adata(seed=0, sparse=False):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS)),
        'int_batch': rng.choice([1, 2, 3], N_CELLS),
        'flag': rng.choice([True, False], N_CELLS),
        'drug': rng.choice(['A', 'B'], N_CELLS),
        'dose': rng.uniform(size=N_CELLS),
    }, index=[f'{seed}_{i}' for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(X) if sparse else X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def read_adata(memory_mode, seed, tmp_path):
    """make_adata(seed) in the layout of memory_mode: sparse for the sparse modes, read backed from tmp_path for backed."""
    adata = make_adata(seed, sparse=memory_mode in ('SparseGPU', 'SparseCPU'))
    if memory_mode == 'backed':
        adata.write_h5ad(tmp_path / f'adata_{seed}.h5ad')
        adata = ad.read_h5ad(tmp_path / f'adata_{seed}.h5ad', backed='r')
    return adata


def make_composer(adata, memory_mode, outdir, **kwargs):
    return Composer(
        adata, memory_mode=memory_mode, compile_model=False, n_top_genes=-1, max_epochs=1, batch_size=64,
        early_stopping=False, deterministic=False, outdir=str(outdir),
        **{'categorical_covariate_keys': ['batch'], **kwargs},
    )


# scVI mode takes the batch as batch_key and any other categorical covariates separately
SCVI_KWARGS = {'mode': 'scvi', 'batch_key': 'batch', 'categorical_covariate_keys': ['drug']}

CONFIGS = [
    # Default model in every memory mode
    pytest.param('GPU', {}, id='GPU', marks=GPU),
    pytest.param('SparseGPU', {}, id='SparseGPU', marks=GPU),
    pytest.param('CPU', {}, id='CPU'),
    pytest.param('SparseCPU', {}, id='SparseCPU'),
    pytest.param('backed', {}, id='backed'),
    # The other distributions and parameterizations (CPU's default is nb, ksi-psi)
    pytest.param('CPU', {'distribution': 'zinb'}, id='CPU-zinb-ksi-psi'),
    pytest.param('CPU', {'parameterization': 'mu-theta'}, id='CPU-nb-mu-theta'),
    pytest.param('CPU', {'distribution': 'zinb', 'parameterization': 'mu-theta'}, id='CPU-zinb-mu-theta'),
    # Covariates stored as numpy and pandas objects
    pytest.param('CPU', {'categorical_covariate_keys': ['int_batch', 'flag', 'batch']}, id='CPU-int-bool-categorical'),
    pytest.param('SparseCPU', {'sparse_continuous_covariate_keys': [('drug', 'dose')]}, id='SparseCPU-sparse-continuous'),
    # scVI mode's model (EtudeSCVI, built from ModuleLists), with the batch as batch_key
    pytest.param('CPU', SCVI_KWARGS, id='CPU-scvi-nb'),
    pytest.param('CPU', {**SCVI_KWARGS, 'distribution': 'zinb'}, id='CPU-scvi-zinb'),
]


@pytest.mark.parametrize('memory_mode, kwargs', CONFIGS)
def test_saved_composer_loads(tmp_path, memory_mode, kwargs):
    adata = read_adata(memory_mode, 0, tmp_path)
    pianist = make_composer(adata, memory_mode, tmp_path, **kwargs)
    pianist.run_pipeline()
    pianist.save(tmp_path / 'pianist.pkl')
    query = read_adata(memory_mode, 1, tmp_path)

    for checkpoint in [None, f'{pianist.checkpoint_path}/model_checkpoint.pt']:
        loaded = Composer.load(tmp_path / 'pianist.pkl', model_checkpoint_path=checkpoint)
        assert type(loaded.model) is type(pianist.model)
        assert loaded.get_latent_representation(adata).shape[0] == N_CELLS
        assert loaded.get_latent_representation(query).shape[0] == N_CELLS


def test_saved_untrained_composer_loads(tmp_path):
    make_composer(make_adata(), 'CPU', tmp_path).save(tmp_path / 'pianist.pkl')

    assert Composer.load(tmp_path / 'pianist.pkl', device='cpu').model is None


# ---------------------------------------------------------------------------
# Guard (fails on master, which has no allowlist, with ImportError): the allowlist holds every layer class and bound
# method of the model, so a new layer type or method fails here, not in users' Composer.load
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('adversarial', [True, False], ids=['adversarial', 'not_adversarial'])
@pytest.mark.parametrize('parameterization', ['ksi-psi', 'mu-theta'])
@pytest.mark.parametrize('distribution', ['nb', 'zinb'])
def test_allowlist_covers_model_layers_and_methods(tmp_path, distribution, parameterization, adversarial):
    from piano.utils.composer import _COMPOSER_PICKLE_GLOBALS, _PICKLED_MODEL_METHODS

    pianist = make_composer(
        make_adata(), 'CPU', tmp_path, distribution=distribution, parameterization=parameterization,
        adversarial=adversarial,
    )
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model(**pianist.model_kwargs)

    # A module pickles as its class, a bound method as getattr(model, method name)
    layer_classes = {(type(module).__module__, type(module).__qualname__) for module in pianist.model.modules()}
    methods = {value.__func__.__name__ for value in vars(pianist.model).values() if inspect.ismethod(value)}
    assert methods
    assert layer_classes <= _COMPOSER_PICKLE_GLOBALS
    assert methods <= _PICKLED_MODEL_METHODS


@pytest.mark.parametrize('distribution', ['nb', 'zinb'])
def test_allowlist_covers_scvi_model_layers_and_methods(tmp_path, distribution):
    """The same guard for scVI mode's model (EtudeSCVI), which stores no bound methods today."""
    from piano.utils.composer import _COMPOSER_PICKLE_GLOBALS, _PICKLED_MODEL_METHODS

    pianist = make_composer(make_adata(), 'CPU', tmp_path, distribution=distribution, **SCVI_KWARGS)
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model(**pianist.model_kwargs)

    layer_classes = {(type(module).__module__, type(module).__qualname__) for module in pianist.model.modules()}
    methods = {value.__func__.__name__ for value in vars(pianist.model).values() if inspect.ismethod(value)}
    assert ('piano.models.scvi_models', 'EtudeSCVI') in layer_classes
    assert layer_classes <= _COMPOSER_PICKLE_GLOBALS
    assert methods <= _PICKLED_MODEL_METHODS
