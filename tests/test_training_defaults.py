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

import inspect
import os
import re
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from piano import Composer
from piano.models.base_models import Etude, EtudeMuTheta

N_GENES = 50
SMALL_BATCH = 32  # Tests on tiny data pass a small batch_size; only the default checks below leave it out
# Repository root, so the subprocesses import this checkout of piano and find run_piano_integration.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI_ENTRY_POINTS = {
    'piano': [sys.executable, '-m', 'piano', '--help'],
    'run_piano_integration': [sys.executable, os.path.join(REPO_ROOT, 'run_piano_integration.py'), '--help'],
}
# The 0.2.0 defaults that both CLIs' --help must show
DEFAULTS = {'n_hidden': 512, 'n_layers': 4, 'max_kld_weight': 0.5, 'batch_size': 512, 'min_adv_weight': 0.5,
            'max_adv_weight': 0.5}
N_ANNEALING_EPOCHS = 200  # Composer default; the CLIs' --n_annealing_epochs already defaulted to 200


# ---------------------------------------------------------------------------
# Helper: small count matrix with a batch covariate
# ---------------------------------------------------------------------------
def make_adata(n_cells=300, seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], n_cells))}, index=[str(i) for i in range(n_cells)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(n_cells, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


# ---------------------------------------------------------------------------
# Composer and model defaults
# ---------------------------------------------------------------------------
def test_default_architecture_is_4_layers_of_512(tmp_path):
    # batch_size is passed so the 300 cells give batches; only the architecture is left to its default
    pianist = Composer(make_adata(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'],
                       n_top_genes=-1, batch_size=SMALL_BATCH, deterministic=False, outdir=str(tmp_path))
    assert (pianist.model_kwargs['n_hidden'], pianist.model_kwargs['n_layers']) == (DEFAULTS['n_hidden'], DEFAULTS['n_layers'])
    model = pianist.prepare_model()
    # Previously 3 hidden layers of 256 units
    assert (model.n_hidden, model.n_layers) == (DEFAULTS['n_hidden'], DEFAULTS['n_layers'])


@pytest.mark.parametrize('model_class', [Etude, EtudeMuTheta])
def test_model_class_defaults_match_the_composer(model_class):
    # Users who build the model directly get the Composer's default architecture (previously 3 x 256)
    parameters = inspect.signature(model_class.__init__).parameters
    assert (parameters['n_hidden'].default, parameters['n_layers'].default) == (DEFAULTS['n_hidden'], DEFAULTS['n_layers'])


def test_saved_composer_keeps_its_architecture(tmp_path):
    # Coverage: a Composer saved with a non-default architecture loads with it, with or without the checkpoint.
    # The pickle and checkpoint are written to this test's temporary directory
    pianist = Composer(make_adata(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'],
                       n_top_genes=-1, n_hidden=256, n_layers=3, batch_size=SMALL_BATCH, max_epochs=1, deterministic=False,
                       outdir=str(tmp_path))
    pianist.run_pipeline()
    pianist.save(tmp_path / 'pianist.pkl')
    checkpoint = f'{pianist.checkpoint_path}/model_checkpoint.pt'
    for loaded in [Composer.load(tmp_path / 'pianist.pkl'),
                   Composer.load(tmp_path / 'pianist.pkl', model_checkpoint_path=checkpoint)]:
        assert (loaded.model.n_hidden, loaded.model.n_layers) == (256, 3)
        assert loaded.get_latent_representation(make_adata()).shape == (300, pianist.model_kwargs['latent_size'])


def test_default_max_kld_weight_is_05(tmp_path):
    # The annealing schedule itself is tested in test_training_schedule.py; here only the default maximum
    pianist = Composer(make_adata(), memory_mode='CPU', compile_model=False, n_top_genes=-1, batch_size=SMALL_BATCH,
                       deterministic=False, outdir=str(tmp_path))
    # Previously 0.25
    assert pianist.max_kld_weight == DEFAULTS['max_kld_weight']


@pytest.mark.parametrize('max_epochs, epoch, expected', [(50, 49, 0.1225), (200, 100, 0.25), (400, 300, 0.5)])
def test_kl_weight_reaches_its_maximum_at_epoch_200_whatever_max_epochs(tmp_path, max_epochs, epoch, expected):
    # Previously the annealing ran over max_epochs, so 50 epochs reached the maximum and 400 epochs reached it at 400
    pianist = Composer(make_adata(), memory_mode='CPU', compile_model=False, n_top_genes=-1, batch_size=SMALL_BATCH,
                       max_epochs=max_epochs, deterministic=False, outdir=str(tmp_path))
    assert pianist.n_annealing_epochs == N_ANNEALING_EPOCHS
    weight = pianist._get_warmup(epoch, 0, 10, pianist.min_kld_weight, pianist.max_kld_weight, pianist.n_annealing_epochs)
    assert weight == pytest.approx(expected)


def test_n_annealing_epochs_none_gives_the_default(tmp_path):
    # None means the default schedule (200 epochs), no longer max_epochs
    pianist = Composer(make_adata(), memory_mode='CPU', compile_model=False, n_top_genes=-1, batch_size=SMALL_BATCH,
                       max_epochs=50, n_annealing_epochs=None, deterministic=False, outdir=str(tmp_path))
    assert pianist.n_annealing_epochs == N_ANNEALING_EPOCHS


def test_default_adversary_weight_is_a_constant_05(tmp_path):
    pianist = Composer(make_adata(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'],
                       n_top_genes=-1, batch_size=SMALL_BATCH, deterministic=False, outdir=str(tmp_path))
    # Previously 1 and 1; equal ends make the annealed weight constant (the schedule is tested in test_training_schedule.py)
    assert (pianist.min_adv_weight, pianist.max_adv_weight) == (DEFAULTS['min_adv_weight'], DEFAULTS['max_adv_weight'])


def test_explicit_adversary_weights_are_kept(tmp_path):
    # Users who relied on the old default pass it explicitly
    pianist = Composer(make_adata(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'],
                       n_top_genes=-1, batch_size=SMALL_BATCH, min_adv_weight=1.0, max_adv_weight=1.0, deterministic=False,
                       outdir=str(tmp_path))
    assert (pianist.min_adv_weight, pianist.max_adv_weight) == (1.0, 1.0)


@pytest.mark.parametrize('early_stopping, expected_epochs', [(None, 4), (True, 2)])
def test_default_trains_every_epoch_without_early_stopping(tmp_path, capsys, early_stopping, expected_epochs):
    # min_delta is out of reach, so early stopping (patience 1) ends training after the first epoch without that gain;
    # previously the default did so too, like early_stopping=True still does
    kwargs = {} if early_stopping is None else {'early_stopping': early_stopping}
    pianist = Composer(make_adata(), memory_mode='CPU', compile_model=False, n_top_genes=-1, batch_size=SMALL_BATCH,
                       max_epochs=4, min_delta=1e9, patience=1, deterministic=False, outdir=str(tmp_path), **kwargs)
    pianist.run_pipeline()
    epoch_lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith('Epoch Total')]
    assert len(epoch_lines) == expected_epochs


def test_default_batch_size_is_512(tmp_path):
    # 600 cells, so one default batch fits
    pianist = Composer(make_adata(n_cells=600), memory_mode='CPU', compile_model=False, n_top_genes=-1, deterministic=False,
                       outdir=str(tmp_path))
    # Previously 128
    assert pianist.batch_size == DEFAULTS['batch_size']


def test_fewer_cells_than_default_batch_raise_at_creation(tmp_path):
    # 300 cells are fewer than one default batch, which drop_last=True drops; previously accepted with batches of 128
    with pytest.raises(ValueError, match=r'Training would have no batches: an epoch draws 300 cells, fewer than batch_size=512'):
        Composer(make_adata(), memory_mode='CPU', compile_model=False, n_top_genes=-1, deterministic=False, outdir=str(tmp_path))


# ---------------------------------------------------------------------------
# Both CLI entry points show the same defaults in their --help
# ---------------------------------------------------------------------------
def get_help_default(help_text, option):
    """Return the 'Default = <value>' of option in an argparse --help text, or None if its help names no default."""
    # Argparse wraps the help over several lines; join them, then take the text from the option to the next option
    text = ' '.join(help_text.split())
    match = re.search(rf'--{option} {option.upper()} (.*?)(?= --|$)', text)
    assert match, f'--{option} is not in the help'
    default = re.search(r'Default = ([^\s.]+(?:\.\d+)?)', match.group(1))
    return None if default is None else default.group(1)


@pytest.fixture(scope='module', params=list(CLI_ENTRY_POINTS))
def cli_help(request):
    # One --help run per entry point; its imports take a few seconds. CUDA is hidden as the help needs no GPU
    result = subprocess.run(CLI_ENTRY_POINTS[request.param], cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize('option', list(DEFAULTS))
def test_cli_help_shows_the_new_defaults(cli_help, option):
    # Previously Default = 256, 3, 0.25, 1.00 and 1.00, and --batch_size named no default (128)
    default = get_help_default(cli_help, option)
    assert default is not None, f'the help of --{option} names no default'
    assert float(default) == DEFAULTS[option]
