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

from contextlib import contextmanager

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from pyro.distributions import ZeroInflatedNegativeBinomial

from piano import Composer
from piano.models.base_models import EtudeMuTheta

N_CELLS = 200
N_GENES = 50
BATCHES = ['a', 'b', 'c']
MIN_GATE_LOGIT = -20  # The clamp floor of the ZINB dropout logits in Etude._decode_latent_zinb
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')


# ---------------------------------------------------------------------------
# Helper: small count matrix with one categorical covariate
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(BATCHES, N_CELLS))}, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.fixture(scope='module', params=[
    ('ksi-psi', True), ('ksi-psi', False), ('mu-theta', True), ('mu-theta', False),
], ids=lambda param: f'{param[0]}-{"adversarial" if param[1] else "no_adversarial"}')
def pianist(request, tmp_path_factory):
    # One CPU epoch of a ZINB model; not deterministic, which needs CUBLAS_WORKSPACE_CONFIG on GPU nodes (review item 16)
    parameterization, adversarial = request.param
    pianist = Composer(
        make_adata(), memory_mode='CPU', compile_model=False, n_top_genes=-1,
        distribution='zinb', parameterization=parameterization, adversarial=adversarial,
        categorical_covariate_keys=['batch'],
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False,
        outdir=str(tmp_path_factory.mktemp('zinb_counterfactuals')),
    )
    pianist.run_pipeline()
    return pianist


@pytest.fixture(scope='module')
def backed_adata(tmp_path_factory):
    # Backed mode reads counts from disk, so the data is written to a pytest temporary folder
    path = tmp_path_factory.mktemp('zinb_backed') / 'query.h5ad'
    make_adata().write_h5ad(path)
    return ad.read_h5ad(path, backed='r')


@contextmanager
def constant_gate(model, gate_logit):
    # Make every dropout logit equal gate_logit (the decoder's NB mean does not depend on it), then restore the weights
    saved = {name: value.clone() for (name, value) in model.decoder_dropouts.state_dict().items()}
    try:
        with torch.no_grad():
            model.decoder_dropouts.weight.zero_()
            model.decoder_dropouts.bias.fill_(gate_logit)
        yield
    finally:
        model.decoder_dropouts.load_state_dict(saved)


def gate_ratio(pianist, gate_logit, **kwargs):
    # Counterfactuals with the gate at gate_logit, divided by those with (almost) no dropout, at the clamp floor
    outputs = {}
    for logit in (gate_logit, MIN_GATE_LOGIT):
        with constant_gate(pianist.model, logit):
            torch.manual_seed(0)  # Same posterior draws for mc_samples > 0
            outputs[logit] = pianist.get_counterfactual(**kwargs)
    return outputs[gate_logit] / outputs[MIN_GATE_LOGIT]


def expected_gate_ratio(gate_logit):
    # (1 - sigmoid(gate_logit)) / (1 - sigmoid(MIN_GATE_LOGIT)), the reference being exact rather than assumed to be 1
    return (torch.sigmoid(torch.tensor(-gate_logit, dtype=torch.float64))
            / torch.sigmoid(torch.tensor(-MIN_GATE_LOGIT, dtype=torch.float64))).item()


# ---------------------------------------------------------------------------
# ZINB counterfactuals are the expected counts (1 - sigmoid(gate)) * mu, not the NB mean mu (review item 40)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('covariates', [None, 'marginal', {'batch': 'b'}], ids=['reconstruction', 'marginal', 'dict'])
def test_zinb_counterfactuals_apply_gate(pianist, covariates):
    # The ratio used to be 1, the gate being ignored
    ratio = gate_ratio(pianist, 2.0, covariates=covariates)
    np.testing.assert_allclose(ratio, expected_gate_ratio(2.0), rtol=1e-5)


def test_zinb_counterfactuals_apply_gate_with_mc_samples(pianist):
    ratio = gate_ratio(pianist, 2.0, covariates='marginal', mc_samples=3)
    np.testing.assert_allclose(ratio, expected_gate_ratio(2.0), rtol=1e-5)


@pytest.mark.parametrize('memory_mode', [
    'CPU',
    'SparseCPU',
    'backed',
    pytest.param('GPU', marks=needs_cuda),
    pytest.param('SparseGPU', marks=needs_cuda),
])
def test_zinb_counterfactuals_apply_gate_in_every_memory_mode(pianist, backed_adata, memory_mode):
    # Query-time memory mode; the model is the same, so this checks that every data path reaches the gate
    adata = backed_adata if memory_mode == 'backed' else make_adata()
    ratio = gate_ratio(pianist, 2.0, adata=adata, covariates='marginal', memory_mode=memory_mode)
    np.testing.assert_allclose(ratio, expected_gate_ratio(2.0), rtol=1e-5)


def zinb_mean(model, posterior_latent, covariates_matrix, library):
    # pyro's ZINB mean for the trained (non-constant) gate, built as in the model's ZINB loss
    x_bar, gate_logits = model._decode_latent(posterior_latent, covariates_matrix)
    if isinstance(model, EtudeMuTheta):
        nb_mu = model._nb_mu(x_bar, library)
        logtheta = torch.clamp(model._nb_logtheta(covariates_matrix), model.min_logit, model.max_logit)
        total_count, logits = torch.exp(logtheta), torch.log(nb_mu) - logtheta
    else:
        nb_mu = model._nb_mu(x_bar, library, covariates_matrix)
        logits = model._nb_psi(covariates_matrix)
        total_count = model._nb_ksi(nb_mu, logits)
    return ZeroInflatedNegativeBinomial(
        total_count=total_count, logits=logits, gate_logits=gate_logits, validate_args=False,
    ).mean


def batch_and_reference_means(model, x_aug, mc_samples):
    # The model's counterfactuals and pyro's ZINB means averaged over the same posterior draws
    with torch.no_grad():
        torch.manual_seed(0)
        output = model.get_batch_counterfactuals(x_aug, mc_samples=mc_samples)

        torch.manual_seed(0)  # Same draws as above
        x_raw, covariates_matrix, _, library = model._parse_augmented_matrix(x_aug)
        posterior_dist = model._encode_latent(model._prepare_encoder_input(x_raw))
        expected = torch.mean(torch.stack([
            zinb_mean(model, posterior_latent, covariates_matrix, library)
            for posterior_latent in model._posterior_latents(posterior_dist, mc_samples)
        ]), dim=0).cpu().numpy()
    return output, expected


@pytest.mark.parametrize('mc_samples', [0, 3])
def test_zinb_batch_counterfactuals_match_pyro_mean(pianist, mc_samples):
    # With mc_samples > 0, each draw's ZINB mean is averaged (the gate is applied per draw, before averaging)
    model = pianist.model
    previously_training = model.training
    model.eval()  # As in get_counterfactuals; restored below, since the model is shared by the module's tests
    x_aug = pianist.train_adataset[torch.arange(N_CELLS)].to(model.device)
    try:
        output, expected = batch_and_reference_means(model, x_aug, mc_samples)
    finally:
        model.train(previously_training)

    np.testing.assert_allclose(output, expected, rtol=1e-5)
