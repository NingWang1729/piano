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
import torch
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50


# ---------------------------------------------------------------------------
# Helper: a Composer trained for one epoch on a small count matrix with a batch covariate
# ---------------------------------------------------------------------------
def train_composer(outdir, parameterization='ksi-psi'):
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    pianist = Composer(
        adata, memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
        parameterization=parameterization, max_epochs=1, batch_size=64, early_stopping=False, deterministic=False,
        outdir=str(outdir),
    )
    pianist.run_pipeline()
    return pianist


@pytest.fixture(scope='module')
def pianist(tmp_path_factory):
    return train_composer(tmp_path_factory.mktemp('piano'))


@pytest.fixture(scope='module', params=['ksi-psi', 'mu-theta'])
def pianist_by_parameterization(request, tmp_path_factory):
    # Etude and EtudeMuTheta each decode counterfactuals
    return train_composer(tmp_path_factory.mktemp(request.param), parameterization=request.param)


def test_latent_is_the_posterior_mean_by_default(pianist):
    # Previously the default was one random draw, so two calls on the same cells differed
    first, second = pianist.get_latent_representation(), pianist.get_latent_representation()
    assert np.array_equal(first, second)


def test_one_posterior_draw_on_request(pianist):
    # mc_samples=1 gives one random draw: different on each call, reproducible with the same seed
    mean = pianist.get_latent_representation()
    torch.manual_seed(0)
    draw = pianist.get_latent_representation(mc_samples=1)
    torch.manual_seed(0)
    assert np.array_equal(draw, pianist.get_latent_representation(mc_samples=1))
    assert not np.array_equal(draw, pianist.get_latent_representation(mc_samples=1))
    assert not np.allclose(draw, mean)


def test_average_of_many_draws_approaches_the_mean(pianist):
    mean = pianist.get_latent_representation()
    torch.manual_seed(0)
    one_draw_error = np.abs(pianist.get_latent_representation(mc_samples=1) - mean).mean()
    many_draws_error = np.abs(pianist.get_latent_representation(mc_samples=400) - mean).mean()
    assert many_draws_error < one_draw_error / 10  # The error of an average of 400 draws shrinks by about sqrt(400) = 20


def test_negative_mc_samples_raise(pianist):
    with pytest.raises(ValueError, match='mc_samples must be 0'):
        pianist.get_latent_representation(mc_samples=-1)


def test_counterfactuals_decode_the_posterior_mean_by_default(pianist_by_parameterization):
    # Previously counterfactuals were decoded from one random posterior draw, so two calls differed
    first, second = pianist_by_parameterization.get_counterfactual(), pianist_by_parameterization.get_counterfactual()
    assert np.array_equal(first, second)


def test_counterfactuals_from_posterior_draws_on_request(pianist_by_parameterization):
    # mc_samples=1 decodes one draw (reproducible with the same seed); many draws average towards the posterior mean's decoding
    mean = pianist_by_parameterization.get_counterfactual()
    torch.manual_seed(0)
    one_draw = pianist_by_parameterization.get_counterfactual(mc_samples=1)
    torch.manual_seed(0)
    assert np.array_equal(one_draw, pianist_by_parameterization.get_counterfactual(mc_samples=1))
    assert not np.allclose(one_draw, mean)
    assert np.isfinite(pianist_by_parameterization.get_counterfactual(mc_samples=20)).all()


def test_negative_mc_samples_raise_for_counterfactuals(pianist):
    with pytest.raises(ValueError, match='mc_samples must be 0'):
        pianist.get_counterfactual(mc_samples=-1)
