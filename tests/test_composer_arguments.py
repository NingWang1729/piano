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
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50


# ---------------------------------------------------------------------------
# Helper: small count matrix with a batch covariate and a sparse continuous covariate (category and value columns)
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS)),
        'cell_type': pd.Categorical(rng.choice(['x', 'y', 'z'], N_CELLS)),
        'score': rng.random(N_CELLS).astype(np.float32),
    }, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(X), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(outdir, memory_mode='CPU', **kwargs):
    kwargs.setdefault('categorical_covariate_keys', ['batch'])
    return Composer(
        make_adata(), memory_mode=memory_mode, compile_model=False, n_top_genes=-1,
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir), **kwargs,
    )


# ---------------------------------------------------------------------------
# sparse_continuous_covariate_keys: only SparseCPU supplies them
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('memory_mode', ['GPU', 'SparseGPU', 'CPU'])
def test_sparse_continuous_covariates_raise_outside_sparse_cpu(memory_mode, tmp_path):
    # Previously failed at the first batch: "mat1 and mat2 shapes cannot be multiplied"
    with pytest.raises(ValueError, match='only supported in SparseCPU memory mode'):
        make_composer(tmp_path, memory_mode, sparse_continuous_covariate_keys=[('cell_type', 'score')])


def test_sparse_continuous_covariates_accepted_in_sparse_cpu(tmp_path):
    pianist = make_composer(tmp_path, 'SparseCPU', sparse_continuous_covariate_keys=[('cell_type', 'score')])
    assert pianist.sparse_continuous_covariate_keys == [('cell_type', 'score')]


def test_counterfactual_in_another_memory_mode_raises_with_sparse_continuous_covariates(tmp_path):
    pianist = make_composer(tmp_path, 'SparseCPU', sparse_continuous_covariate_keys=[('cell_type', 'score')])
    pianist.run_pipeline()

    # The encoder uses the genes only, so latents work in other memory modes
    assert pianist.get_latent_representation(memory_mode='CPU').shape == (N_CELLS, pianist.model_kwargs['latent_size'])

    # The decoder needs the sparse continuous covariates, which only the training memory mode supplies
    assert pianist.get_counterfactual().shape == (N_CELLS, N_GENES)
    # Previously failed with "RuntimeError: The size of tensor a (2) must match the size of tensor b (5)"
    with pytest.raises(ValueError, match='only supported in SparseCPU memory mode, not in CPU mode'):
        pianist.get_counterfactual(memory_mode='CPU')


# ---------------------------------------------------------------------------
# parameterization: unknown values raise instead of training the default model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('parameterization', ['mu_theta', 'MU_THETA', None])
def test_unknown_parameterization_raises_when_composer_is_created(parameterization, tmp_path):
    # Previously only warned in prepare_model and trained the ksi-psi model (None failed in .lower() with an AttributeError)
    with pytest.raises(ValueError, match=rf"parameterization must be one of \('ksi-psi', 'mu-theta'\) \(in any case\), not '{parameterization}'"):
        make_composer(tmp_path, parameterization=parameterization)


@pytest.mark.parametrize('parameterization', ['ksi-psi', 'mu-theta', 'Mu-Theta', 'KSI-PSI'])
def test_known_parameterizations_are_accepted(parameterization, tmp_path):
    pianist = make_composer(tmp_path, parameterization=parameterization)
    assert pianist.parameterization == parameterization.lower()


def test_loading_a_composer_with_an_unknown_parameterization_raises(tmp_path):
    pianist = make_composer(tmp_path)
    pianist.run_pipeline()

    # A Composer pickled before the creation check could hold an unknown value
    pianist.parameterization = 'mu_theta'
    pianist.save(tmp_path / 'composer.pkl')

    # Previously rebuilt the ksi-psi model with only a printed warning
    with pytest.raises(ValueError, match="parameterization must be one of .* not 'mu_theta'"):
        Composer.load(tmp_path / 'composer.pkl', model_checkpoint_path=tmp_path / 'checkpoints' / 'model_checkpoint.pt')


# ---------------------------------------------------------------------------
# adversarial: without categorical covariates the adversary has nothing to predict
# ---------------------------------------------------------------------------
def test_adversarial_without_categorical_covariates_is_turned_off(tmp_path, capsys):
    # Previously trained silently with a 0-output batch classifier: no adversarial loss, but its layers ran every step
    pianist = make_composer(tmp_path, adversarial=True, categorical_covariate_keys=None)
    assert 'Warning: adversarial=True has no effect without categorical_covariate_keys, so adversarial training is turned off' in capsys.readouterr().out
    assert pianist.model_kwargs['adversarial'] is False

    # Training builds the model without an adversary
    pianist.run_pipeline()
    assert pianist.trained_model
    assert pianist.model.batch_classifier is None


@pytest.mark.parametrize('adversarial, categorical_covariate_keys', [(True, ['batch']), (False, None)])
def test_adversarial_setting_kept_when_it_has_effect(adversarial, categorical_covariate_keys, tmp_path, capsys):
    pianist = make_composer(tmp_path, adversarial=adversarial, categorical_covariate_keys=categorical_covariate_keys)
    assert 'adversarial=True has no effect' not in capsys.readouterr().out
    assert pianist.model_kwargs['adversarial'] is adversarial


# ---------------------------------------------------------------------------
# prepare_model: defaults to the Composer's own model kwargs
# ---------------------------------------------------------------------------
def test_prepare_model_without_arguments_uses_composer_settings(tmp_path):
    pianist = make_composer(tmp_path, n_hidden=64, adversarial=False)
    pianist.initialize_features()
    pianist.prepare_data()
    # Previously failed with KeyError: 'adversarial'
    model = pianist.prepare_model()
    assert model.n_hidden == 64 and model.batch_classifier is None

    # One cheap training epoch with the model built this way
    pianist.train()
    assert pianist.trained_model


def test_prepare_model_arguments_override_composer_settings(tmp_path):
    pianist = make_composer(tmp_path, n_hidden=64)
    pianist.initialize_features()
    pianist.prepare_data()
    model = pianist.prepare_model(n_hidden=32)
    assert model.n_hidden == 32 and model.batch_classifier is not None
    assert pianist.model_kwargs['n_hidden'] == 64  # The Composer's settings are left unchanged
