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

import os
import subprocess
import sys
import textwrap

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50
SMALL_MODEL = {'n_hidden': 16, 'n_layers': 1, 'latent_size': 8}  # Fast to train; the matmul precision is what is tested
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
# The precision scope does not depend on the data path, so one CPU and one GPU memory mode are enough
MEMORY_MODES = ['CPU', pytest.param('GPU', marks=GPU)]
# Repository root, so the subprocesses import this checkout of piano and find run_piano_integration.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Helpers: small count matrix with a batch covariate, a small Composer, and a recorder of the matmul precision that
# the model's training step and inference methods run with
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(tmp_path, memory_mode='CPU', **kwargs):
    # compile_model=False: a recorder inside the compiled train step would run only when it is traced
    return Composer(make_adata(), memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=['batch'],
                    n_top_genes=-1, max_epochs=1, batch_size=32, early_stopping=False, deterministic=False,
                    outdir=str(tmp_path), **SMALL_MODEL, **kwargs)


def record_precision(monkeypatch, model, method, records):
    """Wrap model.<method> so each call appends the float32 matmul precision it runs with to records."""
    original = getattr(model, method)

    def recorder(*args, **kwargs):
        records.append(torch.get_float32_matmul_precision())
        return original(*args, **kwargs)

    monkeypatch.setattr(model, method, recorder)


def record_inference(monkeypatch, pianist):
    """Record the precision of the model's latent and counterfactual passes; returns the records by method."""
    records = {'get_latent_representation': [], 'get_counterfactuals': []}
    for method, method_records in records.items():
        record_precision(monkeypatch, pianist.model, method, method_records)
    return records


def run_inference(pianist, adata=None):
    pianist.get_latent_representation(adata)
    pianist.get_counterfactual(adata)


@pytest.fixture(autouse=True)
def restore_matmul_precision():
    # These tests set torch's float32 matmul precision, which is process-wide; restore it for later tests
    precision = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(precision)


# ---------------------------------------------------------------------------
# Training and inference run with the Composer's precision (TF32 by default) and restore the caller's
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
@pytest.mark.parametrize('kwargs, caller, expected', [
    ({}, 'highest', 'high'),
    ({'matmul_precision': 'highest'}, 'high', 'highest'),
], ids=['default_tf32', 'highest'])
def test_training_and_inference_use_the_composers_precision(kwargs, caller, expected, memory_mode, tmp_path, monkeypatch):
    pianist = make_composer(tmp_path, memory_mode, **kwargs)
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model()
    torch.set_float32_matmul_precision(caller)

    train_records = []
    record_precision(monkeypatch, pianist.model, 'training_step', train_records)
    pianist.train()
    # Previously training ran with the caller's precision ('highest' unless the caller changed it)
    assert train_records and set(train_records) == {expected}
    assert torch.get_float32_matmul_precision() == caller

    # Inference on the training data and on query data
    records = record_inference(monkeypatch, pianist)
    for adata in [None, make_adata(seed=1)]:
        run_inference(pianist, adata)
        assert torch.get_float32_matmul_precision() == caller
    assert records == {method: [expected, expected] for method in records}


def test_caller_precision_is_restored_after_errors(tmp_path, monkeypatch):
    # Coverage (master never changes the setting): a caller's own setting, here 'medium', comes back after a Composer
    # call that raises
    pianist = make_composer(tmp_path)
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model()
    torch.set_float32_matmul_precision('medium')

    with pytest.raises(ValueError, match='early_stopping_metric'):
        pianist.train(early_stopping_metric='nope')
    assert torch.get_float32_matmul_precision() == 'medium'

    def failing_step(*args, **kwargs):
        raise RuntimeError('training step failed')

    monkeypatch.setattr(pianist.model, 'training_step', failing_step)
    with pytest.raises(RuntimeError, match='training step failed'):
        pianist.train()
    assert torch.get_float32_matmul_precision() == 'medium'

    # The model passes of get_* raise inside the scope (an invalid covariate key would raise before it is entered)
    def failing_pass(*args, **kwargs):
        raise RuntimeError('model pass failed')

    monkeypatch.setattr(pianist.model, 'get_latent_representation', failing_pass)
    monkeypatch.setattr(pianist.model, 'get_counterfactuals', failing_pass)
    for get in [pianist.get_latent_representation, pianist.get_counterfactual]:
        with pytest.raises(RuntimeError, match='model pass failed'):
            get()
        assert torch.get_float32_matmul_precision() == 'medium'


# ---------------------------------------------------------------------------
# matmul_precision is checked at creation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('value, expected', [('HIGH', 'high'), ('Highest', 'highest'), ('high', 'high')])
def test_matmul_precision_is_accepted_in_any_case(value, expected, tmp_path):
    # Previously an unknown keyword argument (TypeError)
    assert make_composer(tmp_path, matmul_precision=value).matmul_precision == expected


@pytest.mark.parametrize('value', ['fast', 'medium', 1])
def test_invalid_matmul_precision_raises_when_composer_is_created(value, tmp_path):
    # 'medium' allows bf16 matmuls, which change results also on the CPU, so it is not offered
    # Previously an unknown keyword argument (TypeError)
    with pytest.raises(ValueError) as error:
        make_composer(tmp_path, matmul_precision=value)
    # The message names the argument, the value as passed and the valid values
    assert all(text in str(error.value) for text in ['matmul_precision', repr(value), "'highest'", "'high'"])


# ---------------------------------------------------------------------------
# Importing a CLI entry point leaves the process-wide setting alone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('run_entry_point', [
    "runpy.run_module('piano.__main__', run_name='not_main')",
    f"runpy.run_path({os.path.join(REPO_ROOT, 'run_piano_integration.py')!r}, run_name='not_main')",
], ids=['piano', 'run_piano_integration'])
def test_cli_import_leaves_matmul_precision_alone(run_entry_point):
    # A fresh interpreter, so the setting starts at torch's default 'highest'; run_name keeps main() from running
    code = f'''
        import runpy
        import torch
        {run_entry_point}
        print(torch.get_float32_matmul_precision())
    '''
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=REPO_ROOT, capture_output=True, text=True,
                            timeout=600, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stderr
    # Previously both entry points set 'high' for the whole process when imported
    assert result.stdout.strip().splitlines()[-1] == 'highest'


# ---------------------------------------------------------------------------
# Deterministic GPU training stays bit-identical with TF32
# ---------------------------------------------------------------------------
@pytest.fixture
def restore_determinism():
    # deterministic=True turns on deterministic algorithms, which is process-wide; restore the settings for later tests
    algorithms, cudnn = torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic
    yield
    torch.use_deterministic_algorithms(algorithms)
    torch.backends.cudnn.deterministic = cudnn


@GPU
def test_deterministic_gpu_training_with_tf32_repeats_bit_identically(restore_determinism, tmp_path):
    # Coverage: two same-seed runs with the default precision give the same latents. Not compiled: torch.compile's
    # max-autotune needs nvcc, which the Slurm test jobs do not have (no test compiles)
    latents = []
    for run in range(2):
        pianist = Composer(make_adata(), memory_mode='GPU', compile_model=False, categorical_covariate_keys=['batch'],
                           n_top_genes=-1, max_epochs=1, batch_size=32, early_stopping=False, deterministic=True,
                           random_seed=0, outdir=str(tmp_path / f'run_{run}'), **SMALL_MODEL)
        pianist.run_pipeline()
        latents.append(pianist.get_latent_representation())
    np.testing.assert_array_equal(latents[0], latents[1])
