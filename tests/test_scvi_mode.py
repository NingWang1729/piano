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

import contextlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from piano import Composer
from piano.models import EtudeSCVI
from piano.utils.scvi_mode import scvi_kl_weight, scvi_max_epochs, scvi_split_sizes

N_CELLS = 200
N_GENES = 50

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')


# ---------------------------------------------------------------------------
# Helpers: small count matrix with a batch and a donor covariate, and a small scVI-mode Composer
# ---------------------------------------------------------------------------
def make_adata(n_cells=N_CELLS, seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {
            'batch': pd.Categorical(rng.choice(['a', 'b'], n_cells)),
            'donor': pd.Categorical(rng.choice(['d1', 'd2', 'd3'], n_cells)),
            'depth': rng.normal(size=n_cells),
        },
        index=[str(i) for i in range(n_cells)],
    )
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(n_cells, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(tmp_path, adata=None, **kwargs):
    kwargs = {
        'mode': 'scvi', 'batch_key': 'batch', 'categorical_covariate_keys': ['donor'], 'memory_mode': 'CPU',
        'compile_model': False, 'n_hidden': 16, 'latent_size': 4, 'max_epochs': 2, 'batch_size': 64,
        'deterministic': False, 'outdir': str(tmp_path), **kwargs,
    }
    return Composer(make_adata() if adata is None else adata, **kwargs)


# ---------------------------------------------------------------------------
# scVI's defaults in scVI mode, PIANO's in PIANO mode, explicit values in both
# ---------------------------------------------------------------------------
def test_scvi_mode_uses_scvi_defaults(tmp_path):
    pianist = Composer(make_adata(), mode='scvi', batch_key='batch', memory_mode='CPU', outdir=str(tmp_path))
    assert {k: pianist.model_kwargs[k] for k in ('n_hidden', 'n_layers', 'latent_size', 'adversarial')} == {
        'n_hidden': 512, 'n_layers': 4, 'latent_size': 10, 'adversarial': False,  # 4 x 512, not scvi-tools' 1 x 128
    }
    assert (pianist.lr, pianist.weight_decay, pianist.min_kld_weight, pianist.max_kld_weight) == (1e-3, 1e-6, 0.0, 1.0)
    assert (pianist.n_annealing_epochs, pianist.batch_size, pianist.drop_last) == (400, 512, False)  # scvi-tools: 128
    assert (pianist.early_stopping, pianist.min_delta, pianist.patience) == (False, 0.0, 45)
    assert pianist.max_epochs == 400  # scVI's heuristic for 200 cells
    assert pianist.n_top_genes == -1  # scVI uses every gene it is given
    assert pianist.matmul_precision == 'high'  # TF32; scvi-tools trains in FP32


def test_piano_mode_keeps_its_own_defaults(tmp_path):
    # With a categorical covariate, since PIANO turns adversarial training off without one, and more cells than the
    # default batch of 512
    pianist = Composer(make_adata(n_cells=600), categorical_covariate_keys=['batch'], memory_mode='CPU', outdir=str(tmp_path))
    assert {k: pianist.model_kwargs[k] for k in ('n_hidden', 'n_layers', 'latent_size', 'adversarial')} == {
        'n_hidden': 512, 'n_layers': 4, 'latent_size': 32, 'adversarial': True,
    }
    assert (pianist.lr, pianist.weight_decay, pianist.max_kld_weight, pianist.max_epochs) == (2e-4, 0.0, 0.5, 50)
    assert (pianist.n_annealing_epochs, pianist.batch_size, pianist.drop_last) == (200, 512, True)
    assert (pianist.early_stopping, pianist.min_delta, pianist.patience) == (False, 1.0, 5)
    assert pianist.matmul_precision == 'high'


@pytest.mark.parametrize('mode, expected', [('piano', (200, 512, 'high')), ('scvi', (400, 512, 'high'))])
def test_none_gives_the_modes_default(tmp_path, mode, expected):
    # None passed explicitly means the mode's default, as when the setting is left out
    kwargs = {'mode': mode, 'batch_key': 'batch'} if mode == 'scvi' else {'categorical_covariate_keys': ['batch']}
    pianist = Composer(make_adata(n_cells=600), memory_mode='CPU', outdir=str(tmp_path), n_annealing_epochs=None,
                       batch_size=None, matmul_precision=None, **kwargs)
    assert (pianist.n_annealing_epochs, pianist.batch_size, pianist.matmul_precision) == expected


def test_explicit_values_override_scvi_defaults(tmp_path):
    # Also a value equal to PIANO's default, which scVI mode must not replace by scVI's
    pianist = Composer(
        make_adata(), mode='scvi', batch_key='batch', memory_mode='CPU', outdir=str(tmp_path),
        n_hidden=256, n_layers=2, latent_size=8, lr=2e-4, max_epochs=7, early_stopping=True, patience=3, drop_last=True,
        batch_size=64,  # 200 cells: a smaller batch than the default 512, so drop_last=True leaves training batches
    )
    assert (pianist.model_kwargs['n_hidden'], pianist.model_kwargs['n_layers'], pianist.model_kwargs['latent_size']) == (256, 2, 8)
    assert (pianist.lr, pianist.max_epochs, pianist.early_stopping, pianist.patience, pianist.drop_last) == (2e-4, 7, True, 3, True)


# ---------------------------------------------------------------------------
# Training schedule and split as scvi-tools 1.4.2 (scvi-tools is only the reference; skipped without it)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('n_cells', [1, 100, 20_000, 20_001, 535_123, 10_000_000])
def test_max_epochs_heuristic_matches_scvi(n_cells):
    # Both warn when the heuristic gives a single epoch
    model_utils = pytest.importorskip('scvi.model._utils')
    with pytest.warns(UserWarning, match='1') if n_cells == 10_000_000 else contextlib.nullcontext():
        assert scvi_max_epochs(n_cells) == model_utils.get_max_epochs_heuristic(n_cells)


@pytest.mark.parametrize('n_cells', [5, 9, 10, 129, 141, 200, 1000, 1424, 535_123])
@pytest.mark.parametrize('batch_size', [64, 128, 512])
@pytest.mark.parametrize('drop_last', [False, True])
def test_split_sizes_match_scvi(n_cells, batch_size, drop_last):
    # 90% of cells train; without drop_last, a last training batch of 1 or 2 cells moves to the validation set (or is
    # left out when there is no validation set)
    splitting = pytest.importorskip('scvi.dataloaders._data_splitting')
    expected = splitting.validate_data_split(n_cells, 0.9, None, batch_size, drop_last, True)
    assert scvi_split_sizes(n_cells, batch_size, drop_last) == tuple(expected)


@pytest.mark.parametrize('n_annealing_epochs', [400, 5])
def test_kl_weight_matches_scvi(n_annealing_epochs):
    # Constant within an epoch, linear in the epoch index, then the maximum
    trainingplans = pytest.importorskip('scvi.train._trainingplans')
    for epoch in range(12):
        expected = trainingplans._compute_kl_weight(epoch, 0, n_annealing_epochs, None, 1.0, 0.0)
        assert scvi_kl_weight(epoch, n_annealing_epochs, 0.0, 1.0) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Options that cannot apply in scVI mode raise when the Composer is created
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kwargs, error, match', [
    ({'batch_key': None}, ValueError, 'batch_key'),
    ({'categorical_covariate_keys': ['batch', 'donor']}, ValueError, 'batch_key'),
    ({'adversarial': True}, ValueError, 'adversarial'),
    ({'parameterization': 'mu-theta'}, ValueError, 'parameterization'),
    ({'sparse_continuous_covariate_keys': [('donor', 'depth')]}, NotImplementedError, 'sparse_continuous'),
    ({'stratify_column': 'donor', 'samples_per_class': 10}, NotImplementedError, 'stratif'),
    ({'batchnorm_eps': 1e-3}, ValueError, 'batchnorm_eps'),
    ({'max_adv_weight': 2.0}, ValueError, 'max_adv_weight'),
    ({'decoder_dropout_rate': 0.2}, ValueError, 'decoder_dropout_rate'),
    ({'decoder_dropout_rate': None}, ValueError, 'decoder_dropout_rate'),  # None (same as dropout_rate) is not scVI's
    ({'mode': 'scVI-tools'}, ValueError, 'mode'),
])
def test_options_that_cannot_apply_raise(kwargs, error, match, tmp_path):
    with pytest.raises(error, match=match):
        make_composer(tmp_path, **kwargs)


def test_early_stopping_without_validation_cells_raises(tmp_path):
    # 5 cells: scVI's 90% split leaves none for validation, which early stopping scores
    with pytest.raises(ValueError, match='validation'):
        make_composer(tmp_path, adata=make_adata(n_cells=5), early_stopping=True, batch_size=2)


def test_batch_key_in_piano_mode_warns(tmp_path):
    with pytest.warns(UserWarning, match='batch_key'):
        Composer(make_adata(), batch_key='batch', categorical_covariate_keys=['batch'], memory_mode='CPU', batch_size=64,
                 outdir=str(tmp_path))


def test_counterfactuals_not_available_in_scvi_mode(tmp_path):
    pianist = make_composer(tmp_path)
    pianist.run_pipeline()
    with pytest.raises(NotImplementedError, match='scVI mode'):
        pianist.get_counterfactual()


def test_early_stopping_metric_cannot_be_chosen_in_scvi_mode(tmp_path):
    # scVI's early stopping always monitors the validation ELBO
    pianist = make_composer(tmp_path)
    with pytest.raises(ValueError, match='early_stopping_metric'):
        pianist.train(early_stopping_metric='nll')


# ---------------------------------------------------------------------------
# Training in scVI mode
# ---------------------------------------------------------------------------
def test_model_is_etude_scvi_with_batch_first(tmp_path):
    pianist = make_composer(tmp_path)
    model = pianist.prepare_model()
    assert isinstance(model, EtudeSCVI)
    assert (model.n_batch, model.n_cats_per_cov) == (2, (3,))
    assert (model.n_hidden, model.latent_size) == (16, 4)
    assert model.distribution == 'nb'  # scVI mode's default likelihood


def adata_with_genes_expressed(seed=0):
    """make_adata's cells with gene.counts, the number of genes expressed in each cell (a raw continuous covariate)."""
    adata = make_adata(seed=seed)
    adata.obs['gene.counts'] = np.asarray((adata.X > 0).sum(axis=1)).ravel().astype(np.float32)
    return adata


@pytest.mark.parametrize('memory_mode', ['CPU', 'GPU', 'SparseCPU'])
@pytest.mark.parametrize('distribution', ['nb', 'zinb'])
def test_end_to_end_with_continuous_covariates(tmp_path, memory_mode, distribution):
    if memory_mode == 'GPU' and not torch.cuda.is_available():
        pytest.skip('needs a CUDA GPU')
    pianist = make_composer(tmp_path, adata=adata_with_genes_expressed(), continuous_covariate_keys=['gene.counts'],
                            memory_mode=memory_mode, distribution=distribution, early_stopping=True)
    pianist.run_pipeline()
    assert pianist.model.n_continuous_cov == 1
    latent = pianist.get_latent_representation()
    assert latent.shape == (N_CELLS, 4) and np.isfinite(latent).all()


def test_continuous_covariates_reach_the_decoder(tmp_path):
    # The augmented matrix ends with gene.counts' raw values, which the training step reads: other values, another loss
    adata = adata_with_genes_expressed()
    pianist = make_composer(tmp_path, adata=adata, continuous_covariate_keys=['gene.counts'])
    x_aug = pianist.prepare_data()[np.arange(8)]
    model = pianist.prepare_model()
    assert torch.equal(x_aug[:, -1].cpu(), torch.from_numpy(adata.obs['gene.counts'].to_numpy()[:8]))

    model.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        loss = model.training_step(x_aug.to(model.device), torch.tensor(1.0))['loss']
        shifted = x_aug.clone()
        shifted[:, -1] += 10.0
        torch.manual_seed(0)
        shifted_loss = model.training_step(shifted.to(model.device), torch.tensor(1.0))['loss']
    assert not torch.isclose(loss, shifted_loss)


def test_end_to_end_with_zinb(tmp_path):
    pianist = make_composer(tmp_path, distribution='zinb')
    pianist.run_pipeline()
    assert pianist.model.distribution == 'zinb'
    latent = pianist.get_latent_representation()
    assert latent.shape == (N_CELLS, 4) and np.isfinite(latent).all()


def test_trains_on_scvi_split(tmp_path):
    pianist = make_composer(tmp_path, random_seed=3)
    pianist.run_pipeline()
    train, validation = pianist.train_indices, pianist.validation_indices
    assert (len(train), len(validation)) == (180, 20)
    assert sorted(np.concatenate([train, validation]).tolist()) == list(range(N_CELLS))

    # The split depends only on the random seed
    other = make_composer(tmp_path / 'other', random_seed=3)
    other.run_pipeline()
    assert (other.train_indices == train).all() and (other.validation_indices == validation).all()


@pytest.mark.parametrize('memory_mode', [
    'CPU', 'SparseCPU', 'backed',
    pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU),
])
def test_end_to_end_with_early_stopping(tmp_path, memory_mode):
    # Every memory mode, since scVI's split adds a dataset subset; backed mode reads an h5ad written to tmp_path
    adata = make_adata()
    if memory_mode == 'backed':
        adata.write_h5ad(tmp_path / 'counts.h5ad')
        adata = ad.read_h5ad(tmp_path / 'counts.h5ad', backed='r')
    pianist = make_composer(tmp_path, adata=adata, memory_mode=memory_mode, early_stopping=True)
    pianist.run_pipeline()
    latent = pianist.get_latent_representation()
    assert latent.shape == (N_CELLS, 4) and np.isfinite(latent).all()


def test_epochs_heuristic_only_without_max_epochs(tmp_path, monkeypatch):
    # The heuristic can warn about a 1-epoch default, which would be wrong when max_epochs is given
    def fail(n_cells):
        raise AssertionError('scvi_max_epochs called although max_epochs was given')
    monkeypatch.setattr('piano.utils.composer.scvi_max_epochs', fail)
    assert make_composer(tmp_path, max_epochs=3).max_epochs == 3


def test_early_stopping_stops_after_patience(tmp_path):
    # The first epoch improves on no previous score; with an unreachable min_delta no later one does
    pianist = make_composer(
        tmp_path, max_epochs=10, early_stopping=True, patience=2, min_delta=1e9, checkpoint_every_n_epochs=1,
    )
    pianist.run_pipeline()
    # Epoch checkpoints, after the initial weights (epoch -1): epoch 0 improves, 1 and 2 do not, so training stops after 2
    checkpoints = sorted(p.name for p in (tmp_path / 'checkpoints').glob('model_epoch=*.pt'))
    assert checkpoints == ['model_epoch=-1.pt', 'model_epoch=0.pt', 'model_epoch=1.pt', 'model_epoch=2.pt']


def test_keeps_last_epoch_weights(tmp_path):
    # As scvi-tools, the trained model is the last epoch's, not the best epoch's
    pianist = make_composer(tmp_path, max_epochs=3, checkpoint_every_n_epochs=1)
    pianist.run_pipeline()
    last = torch.load(tmp_path / 'checkpoints' / 'model_epoch=2.pt')
    final = torch.load(tmp_path / 'checkpoints' / 'model_checkpoint.pt')
    assert final.keys() == last.keys() and all(torch.equal(final[k], last[k]) for k in final)


def test_optimizer_has_scvi_settings(tmp_path):
    pianist = make_composer(tmp_path, lr=3e-3)
    pianist.prepare_model()
    optimizer = pianist._scvi_optimizer()
    assert isinstance(optimizer, torch.optim.Adam) and not isinstance(optimizer, torch.optim.AdamW)
    group = optimizer.param_groups[0]
    assert (group['lr'], group['eps'], group['weight_decay']) == (3e-3, 0.01, 1e-6)


@pytest.mark.parametrize('matmul_precision', [None, 'highest'])
def test_training_runs_at_the_composers_matmul_precision(tmp_path, matmul_precision, monkeypatch):
    # scVI training, like PIANO's, runs inside the Composer's precision scope (TF32 by default, FP32 when asked); the
    # caller's setting is restored afterwards
    pianist = make_composer(tmp_path, matmul_precision=matmul_precision)
    pianist.prepare_data()
    pianist.prepare_model()
    seen = []
    monkeypatch.setattr(pianist, '_train_scvi', lambda: seen.append(torch.get_float32_matmul_precision()))
    # A caller setting different from both choices, so the scope is visible either way
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('medium')
    try:
        pianist.train()
        assert seen == [matmul_precision or 'high']
        assert torch.get_float32_matmul_precision() == 'medium'
    finally:
        torch.set_float32_matmul_precision(previous)


def test_save_load_round_trip_with_continuous_covariates(tmp_path):
    pianist = make_composer(tmp_path, adata=adata_with_genes_expressed(), continuous_covariate_keys=['gene.counts'])
    pianist.run_pipeline()
    pianist.save(tmp_path / 'pianist.pkl')
    loaded = Composer.load(tmp_path / 'pianist.pkl', model_checkpoint_path=f'{pianist.checkpoint_path}/model_checkpoint.pt')
    assert loaded.model.n_continuous_cov == 1
    state, loaded_state = pianist.model.state_dict(), loaded.model.state_dict()
    assert state.keys() == loaded_state.keys() and all(torch.equal(state[k].cpu(), loaded_state[k].cpu()) for k in state)


@pytest.mark.parametrize('distribution', ['nb', 'zinb'])
def test_save_load_round_trip(tmp_path, distribution):
    pianist = make_composer(tmp_path, distribution=distribution)
    pianist.run_pipeline()
    latent = pianist.get_latent_representation()
    pianist.save(tmp_path / 'pianist.pkl')
    loaded = Composer.load(tmp_path / 'pianist.pkl', model_checkpoint_path=f'{pianist.checkpoint_path}/model_checkpoint.pt')
    assert isinstance(loaded.model, EtudeSCVI) and loaded.model.distribution == distribution
    assert np.allclose(loaded.get_latent_representation(make_adata()), latent, atol=1e-5)


@GPU
def test_compiled_gpu_training(tmp_path):
    # PIANO's fast path: data on the GPU and a compiled training step, with scVI's last partial batch (180 = 2 x 64 + 52)
    pianist = make_composer(tmp_path, memory_mode='GPU', compile_model=True, early_stopping=True)
    pianist.run_pipeline()
    latent = pianist.get_latent_representation()
    assert latent.shape == (N_CELLS, 4) and np.isfinite(latent).all()
