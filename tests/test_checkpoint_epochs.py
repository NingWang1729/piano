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

import json
import os
import re
import subprocess
import sys
import textwrap
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 30
MODES = ['piano', 'scvi']
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
# Repository root, so the subprocesses import this checkout of piano and find run_piano_integration.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Helpers: small count matrix with a batch covariate, a small Composer in either mode, and checkpoint utilities
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(tmp_path, mode='piano', **kwargs):
    # CPU without torch.compile, so two runs with the same seed in this process train identically. n_annealing_epochs is
    # set (and short), so the KL weight changes from epoch to epoch and does not depend on max_epochs
    mode_kwargs = {'mode': 'scvi', 'batch_key': 'batch'} if mode == 'scvi' else {'categorical_covariate_keys': ['batch']}
    kwargs = {
        'memory_mode': 'CPU', 'compile_model': False, 'n_top_genes': -1, 'n_hidden': 16, 'n_layers': 1, 'latent_size': 4,
        'batch_size': 64, 'max_epochs': 3, 'n_annealing_epochs': 3, 'early_stopping': False, 'deterministic': False,
        'random_seed': 0, 'outdir': str(tmp_path), **mode_kwargs, **kwargs,
    }
    return Composer(make_adata(), **kwargs)


def train(tmp_path, mode='piano', **kwargs):
    pianist = make_composer(tmp_path, mode, **kwargs)
    pianist.run_pipeline()
    return pianist


def checkpoint(tmp_path, name):
    return torch.load(f'{tmp_path}/checkpoints/{name}', map_location='cpu', weights_only=True)


def saved_weight_files(tmp_path):
    return {name for name in os.listdir(f'{tmp_path}/checkpoints') if name.endswith('.pt')}


def same_weights(a, b):
    assert a.keys() == b.keys()
    return all(torch.equal(a[name].cpu(), b[name].cpu()) for name in a)


def model_weights(pianist):
    return {name: weights.detach().cpu().clone() for name, weights in pianist.model.state_dict().items()}


def latent_of_checkpoint(pianist, path):
    pianist.load_model(path)
    return pianist.get_latent_representation()  # The posterior mean, which is deterministic


@pytest.fixture
def restore_determinism():
    # deterministic=True turns deterministic algorithms on, which is process-wide; restore the settings for later tests
    algorithms, cudnn = torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic
    yield
    torch.use_deterministic_algorithms(algorithms)
    torch.backends.cudnn.deterministic = cudnn


# ---------------------------------------------------------------------------
# Checking the values when the Composer is created
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('checkpoint_epochs', [[0], [4], [1, 1], [1.5], ['2']],
                         ids=['zero', 'above_max_epochs', 'duplicate', 'float', 'string'])
def test_invalid_checkpoint_epochs_raise_when_composer_is_created(tmp_path, mode, checkpoint_epochs):
    # max_epochs=3: values must be integers from 1 to 3, each at most once
    with pytest.raises(ValueError, match='checkpoint_epochs'):
        make_composer(tmp_path, mode, checkpoint_epochs=checkpoint_epochs)


@pytest.mark.parametrize('n_annealing_epochs', [None, 3])
@pytest.mark.parametrize('mode', MODES)
def test_no_warning_at_creation(tmp_path, mode, n_annealing_epochs):
    # The default KL annealing schedule is a fixed number of epochs, so the epoch-N weights match an N-epoch run with
    # either setting: nothing to warn about, and the default (None) must not warn on every run
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        make_composer(tmp_path, mode, checkpoint_epochs=[1], n_annealing_epochs=n_annealing_epochs)
    assert not [w for w in caught if 'checkpoint_epochs' in str(w.message) or 'n_annealing_epochs' in str(w.message)]


# ---------------------------------------------------------------------------
# Which files training writes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('mode', MODES)
def test_only_the_listed_epochs_are_saved(tmp_path, mode):
    # max_epochs itself is allowed: it saves the last epoch's weights
    train(tmp_path, mode, checkpoint_epochs=[1, 3])
    assert saved_weight_files(tmp_path) == {
        'model_epochs=1.pt', 'model_epochs=3.pt', 'model_best.pt', 'model_last.pt', 'model_checkpoint.pt',
    }


@pytest.mark.parametrize('mode', MODES)
def test_without_checkpoint_epochs_only_the_last_and_final_models_are_saved(tmp_path, mode):
    # model_last.pt is always written; model_best.pt and model_epochs=N.pt only with checkpoint_epochs
    train(tmp_path, mode)
    assert saved_weight_files(tmp_path) == {'model_last.pt', 'model_checkpoint.pt'}


@pytest.mark.parametrize('mode', MODES)
def test_checkpoint_epochs_do_not_change_training(tmp_path, mode):
    # In scVI mode the validation ELBO is then scored every epoch, which must not change the training either
    plain = model_weights(train(tmp_path / 'plain', mode))
    with_checkpoints = model_weights(train(tmp_path / 'with_checkpoints', mode, checkpoint_epochs=[1, 2]))
    assert same_weights(plain, with_checkpoints)


@GPU
def test_scvi_mode_checkpoint_epochs_do_not_change_gpu_training(tmp_path, restore_determinism):
    # On the GPU the posterior samples of the validation ELBO draw from the CUDA generator, which training draws from too
    gpu = {'memory_mode': 'GPU', 'deterministic': True}
    plain = model_weights(train(tmp_path / 'plain', 'scvi', **gpu))
    with_checkpoints = model_weights(train(tmp_path / 'with_checkpoints', 'scvi', checkpoint_epochs=[1, 2], **gpu))
    assert same_weights(plain, with_checkpoints)


# ---------------------------------------------------------------------------
# Listed epochs that training does not reach
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('mode', MODES)
def test_listed_epoch_after_early_stopping_warns_and_is_not_saved(tmp_path, mode):
    # min_delta is out of reach, so with patience=1 training stops after its 2nd epoch of 5, before epoch 4
    with pytest.warns(UserWarning, match=r'checkpoint_epochs \[4\] were not saved.*early stopping'):
        train(tmp_path, mode, max_epochs=5, early_stopping=True, min_delta=1e9, patience=1, checkpoint_epochs=[1, 4])
    assert 'model_epochs=1.pt' in saved_weight_files(tmp_path)
    assert 'model_epochs=4.pt' not in saved_weight_files(tmp_path)


@pytest.mark.parametrize('mode', MODES)
def test_listed_epoch_above_a_lowered_max_epochs_warns(tmp_path, mode):
    # The values are checked against max_epochs when the Composer is created; lowering it afterwards skips epoch 3
    pianist = make_composer(tmp_path, mode, checkpoint_epochs=[1, 3])
    pianist.max_epochs = 2
    with pytest.warns(UserWarning, match=r'checkpoint_epochs \[3\] were not saved.*max_epochs=2'):
        pianist.run_pipeline()
    assert 'model_epochs=3.pt' not in saved_weight_files(tmp_path)


@pytest.mark.parametrize('mode', MODES)
def test_no_warning_when_every_listed_epoch_is_saved(tmp_path, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        train(tmp_path, mode, checkpoint_epochs=[1, 3])
    assert not [w for w in caught if 'were not saved' in str(w.message)]


# ---------------------------------------------------------------------------
# Training on from model_last.pt, as the README shows: a new Composer with a new outdir
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('mode', MODES)
def test_continue_training_from_model_last_in_a_new_outdir(tmp_path, mode):
    first = tmp_path / 'first_200_epochs'
    train(first, mode, max_epochs=2, checkpoint_epochs=[2])
    first_files = {name: checkpoint(first, name) for name in saved_weight_files(first)}

    # save_initial_weights writes the weights training starts from to model_epoch=-1.pt
    more = tmp_path / 'more_epochs'
    pianist = make_composer(more, mode, max_epochs=1, save_initial_weights=True)
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model(**pianist.model_kwargs)
    pianist.load_model(f'{first}/checkpoints/model_last.pt')
    pianist.train()

    # Training started from the loaded weights and changed them; the first run's files are untouched
    assert same_weights(checkpoint(more, 'model_epoch=-1.pt'), first_files['model_last.pt'])
    assert not same_weights(checkpoint(more, 'model_last.pt'), first_files['model_last.pt'])
    assert saved_weight_files(first) == first_files.keys()
    assert all(same_weights(checkpoint(first, name), weights) for name, weights in first_files.items())


# ---------------------------------------------------------------------------
# Log messages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('mode', MODES)
def test_model_last_message_counts_epochs_trained(tmp_path, mode, capsys):
    # Epochs trained (1-based), as model_epochs=<N>.pt and best_epoch count them
    train(tmp_path, mode, checkpoint_epochs=[3])
    out = capsys.readouterr().out
    assert re.search(r'Model of the last epoch, after 3 epochs, saved to .*model_last\.pt', out)


# ---------------------------------------------------------------------------
# What the checkpoints hold
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('mode', MODES)
def test_epoch_n_checkpoint_gives_the_latent_of_an_n_epoch_run(tmp_path, mode):
    # The 2-epoch run's last epoch comes from checkpoint_every_n_epochs (model_epoch=1.pt, 0-based), which exists already,
    # since in PIANO mode the run's final model is its best epoch, not necessarily its last
    long_run = train(tmp_path / 'long', mode, max_epochs=3, checkpoint_epochs=[2])
    short_run = train(tmp_path / 'short', mode, max_epochs=2, checkpoint_every_n_epochs=2)
    latent = latent_of_checkpoint(long_run, f'{tmp_path}/long/checkpoints/model_epochs=2.pt')
    expected = latent_of_checkpoint(short_run, f'{tmp_path}/short/checkpoints/model_epoch=1.pt')
    np.testing.assert_array_equal(latent, expected)


def test_piano_mode_best_model_is_the_final_saved_model(tmp_path):
    pianist = make_composer(tmp_path, checkpoint_epochs=[1], checkpoint_every_n_epochs=1)
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model(**pianist.model_kwargs)

    # Make every epoch after the first look worse in the loss that picks the best epoch ('ref'), so the best epoch is the
    # first, not the last; training itself follows the 'total' loss and is unchanged (as in test_training_checkpoints.py)
    finished_epochs = 0
    print_epoch_losses = pianist._print_epoch_losses
    def count_finished_epochs(*args, **kwargs):
        nonlocal finished_epochs
        finished_epochs += 1
        return print_epoch_losses(*args, **kwargs)
    pianist._print_epoch_losses = count_finished_epochs

    training_step = pianist.model.training_step
    def training_step_with_later_epochs_worse(batch, kld_weight, adv_weight):
        losses = training_step(batch, kld_weight, adv_weight)
        if finished_epochs > 0:
            losses['ref'] = losses['ref'] + 1e6
        return losses
    pianist.model.training_step = training_step_with_later_epochs_worse

    pianist.train()
    # best_epoch counts epochs trained (1-based)
    assert pianist.best_epoch == 1
    best = checkpoint(tmp_path, 'model_best.pt')
    assert same_weights(best, checkpoint(tmp_path, 'model_checkpoint.pt'))
    assert same_weights(best, checkpoint(tmp_path, 'model_epochs=1.pt'))
    # The last epoch's weights differ from the best: model_last.pt holds the last (0-based model_epoch=2.pt)
    last = checkpoint(tmp_path, 'model_last.pt')
    assert same_weights(last, checkpoint(tmp_path, 'model_epoch=2.pt'))
    assert not same_weights(last, best)


def test_scvi_mode_best_model_is_the_lowest_validation_elbo(tmp_path, capsys):
    # early_stopping is off, so the validation ELBO is scored every epoch only because checkpoint_epochs is set
    pianist = make_composer(tmp_path, 'scvi', checkpoint_epochs=[1, 2, 3])
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model(**pianist.model_kwargs)

    # Make the validation ELBO of every epoch after the first much worse, so the best epoch is the first, not the last
    # scVI mode keeps; only the model's eval-mode (validation) losses change, so training is unchanged
    finished_epochs = 0
    train_scvi_epoch = pianist._train_scvi_epoch
    def count_finished_epochs(*args, **kwargs):
        nonlocal finished_epochs
        terms = train_scvi_epoch(*args, **kwargs)
        finished_epochs += 1
        return terms
    pianist._train_scvi_epoch = count_finished_epochs

    training_step = pianist.model.training_step
    def training_step_with_later_validation_worse(batch, kld_weight, adv_weight):
        losses = training_step(batch, kld_weight, adv_weight)
        if not pianist.model.training and finished_epochs > 1:
            losses['reconstruction'] = losses['reconstruction'] + 1e6
        return losses
    pianist.model.training_step = training_step_with_later_validation_worse

    pianist.train()
    elbos = [float(value) for value in re.findall(r'Validation ELBO: ([-\d.e+]+)', capsys.readouterr().out)]
    assert len(elbos) == 3, 'the validation ELBO is printed every epoch'
    assert int(np.argmin(elbos)) + 1 == pianist.best_epoch == 1
    best = checkpoint(tmp_path, 'model_best.pt')
    assert same_weights(best, checkpoint(tmp_path, 'model_epochs=1.pt'))
    # scVI mode still keeps the last epoch as its model, which model_last.pt holds
    last = checkpoint(tmp_path, 'model_last.pt')
    assert same_weights(last, model_weights(pianist))
    assert same_weights(last, checkpoint(tmp_path, 'model_epochs=3.pt'))
    assert not same_weights(last, best)


def test_scvi_mode_without_a_finite_validation_elbo_warns_and_skips_model_best(tmp_path):
    # A validation ELBO that is NaN in every epoch (training itself finite) leaves no best epoch: training still finishes
    # and saves its model, without model_best.pt
    pianist = make_composer(tmp_path, 'scvi', checkpoint_epochs=[1])
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model(**pianist.model_kwargs)

    training_step = pianist.model.training_step
    def training_step_with_nan_validation(batch, kld_weight, adv_weight):
        losses = training_step(batch, kld_weight, adv_weight)
        if not pianist.model.training:
            losses['reconstruction'] = losses['reconstruction'] + float('nan')
        return losses
    pianist.model.training_step = training_step_with_nan_validation

    with pytest.warns(UserWarning, match='model_best.pt'):
        pianist.train()
    assert pianist.trained_model
    assert pianist.best_epoch is None
    assert saved_weight_files(tmp_path) == {'model_epochs=1.pt', 'model_last.pt', 'model_checkpoint.pt'}


@pytest.mark.parametrize('mode', MODES)
def test_model_last_is_the_epoch_early_stopping_stopped_at(tmp_path, mode):
    # min_delta is out of reach, so with patience=1 training stops after its 2nd epoch of 5
    train(tmp_path, mode, max_epochs=5, early_stopping=True, min_delta=1e9, patience=1, checkpoint_every_n_epochs=1)
    assert 'model_epoch=1.pt' in saved_weight_files(tmp_path)
    assert 'model_epoch=2.pt' not in saved_weight_files(tmp_path)
    assert same_weights(checkpoint(tmp_path, 'model_last.pt'), checkpoint(tmp_path, 'model_epoch=1.pt'))


# ---------------------------------------------------------------------------
# End to end: train, then load an epoch-N checkpoint and embed
# ---------------------------------------------------------------------------
def test_end_to_end_with_checkpoint_epochs(tmp_path):
    pianist = train(tmp_path, checkpoint_epochs=[1, 2], max_epochs=2)
    for name in ['model_epochs=1.pt', 'model_epochs=2.pt', 'model_best.pt', 'model_last.pt']:
        latent = latent_of_checkpoint(pianist, f'{tmp_path}/checkpoints/{name}')
        assert latent.shape == (N_CELLS, 4)
        assert np.isfinite(latent).all()


# ---------------------------------------------------------------------------
# Command-line flag: --checkpoint_epochs reaches the Composer in both entry points
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('run_entry_point', [
    "runpy.run_module('piano.__main__', run_name='__main__')",
    f"runpy.run_path({os.path.join(REPO_ROOT, 'run_piano_integration.py')!r}, run_name='__main__')",
], ids=['piano', 'run_piano_integration'])
@pytest.mark.parametrize('flag, expected', [(['--checkpoint_epochs', '20', '50'], [20, 50]), ([], None)],
                         ids=['given', 'default'])
def test_cli_passes_checkpoint_epochs_to_the_composer(tmp_path, run_entry_point, flag, expected):
    # A fresh interpreter in which piano.Composer records its arguments and stops the script, so nothing is trained
    path = tmp_path / 'adata.h5ad'
    make_adata().write_h5ad(path)
    argv = ['entry_point', '--adata_train_list', str(path), '--adata_valid', str(path), '--outdir', str(tmp_path),
            '--memory_mode', 'CPU', '--n_top_genes', '-1', '--batch_key', 'batch', '--categorical_covariate_keys', 'batch',
            *flag]
    code = f'''
        import json, runpy, sys
        import piano

        class ComposerCalled(Exception):
            pass

        def record_composer(*args, **kwargs):
            print(json.dumps(kwargs.get('checkpoint_epochs', 'not passed')))
            raise ComposerCalled

        piano.Composer = record_composer
        sys.argv = {argv!r}
        try:
            {run_entry_point}
        except ComposerCalled:
            pass
    '''
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=REPO_ROOT, capture_output=True, text=True,
                            timeout=600, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == expected
