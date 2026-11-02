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

import pytest
import torch
import torch.nn.functional as F

from piano.models import EtudeSCVI

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CELLS = 64
N_GENES = 30
ARCH = {'n_hidden': 16, 'latent_size': 4}

# Float tolerance: the same model computed with a different order of operations
TOLERANCE = {'rtol': 1e-5, 'atol': 1e-5}

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')

# (n_batch, n_cats_per_cov, n_layers): one hidden layer (scVI's default), three (a deeper network), an extra categorical
# covariate, and a single batch (which scVI leaves out of the decoder, as it carries no information)
CONFIGS = [(3, (), 1), (3, (), 3), (3, (5,), 3), (1, (), 2)]


# ---------------------------------------------------------------------------
# Helpers: scVI's own VAE and EtudeSCVI with the same weights, and one minibatch in both input formats
#
# scvi-tools is only a reference here (PIANO's optional scvi-tools extra); tests that need it skip without it
# ---------------------------------------------------------------------------
def scvi_vae(n_batch, n_cats_per_cov, n_layers, dropout_rate=0.1):
    vae_module = pytest.importorskip('scvi.module')
    return vae_module.VAE(
        N_GENES, n_batch=n_batch, n_labels=1, n_cats_per_cov=list(n_cats_per_cov) or None, n_layers=n_layers,
        n_hidden=ARCH['n_hidden'], n_latent=ARCH['latent_size'], dropout_rate=dropout_rate, gene_likelihood='nb',
        dispersion='gene-batch',
    )


def matched_pair(n_batch, n_cats_per_cov, n_layers, dropout_rate=0.1):
    """scVI's VAE and an EtudeSCVI that loaded its weights."""
    torch.manual_seed(0)
    reference = scvi_vae(n_batch, n_cats_per_cov, n_layers, dropout_rate)
    model = EtudeSCVI(N_GENES, n_batch, n_cats_per_cov, n_layers=n_layers, dropout_rate=dropout_rate, **ARCH)
    model.load_scvi_state_dict(reference.state_dict())
    return reference, model


def make_batch(n_batch, n_cats_per_cov, device='cpu', seed=1):
    """scVI's tensors dict and EtudeSCVI's inputs (counts, batch one-hot, decoder covariates) for the same cells."""
    g = torch.Generator().manual_seed(seed)
    x = torch.poisson(torch.full((N_CELLS, N_GENES), 3.0), generator=g)
    batch = torch.randint(0, n_batch, (N_CELLS, 1), generator=g)
    cats = [torch.randint(0, n, (N_CELLS, 1), generator=g) for n in n_cats_per_cov]
    tensors = {'X': x, 'batch': batch, 'labels': torch.zeros(N_CELLS, 1, dtype=torch.long)}
    if cats:
        tensors['extra_categorical_covs'] = torch.cat(cats, dim=1)

    # One-hot covariates in scVI's decoder order: batch, then each categorical covariate; single categories left out
    batch_one_hot = F.one_hot(batch.squeeze(-1), n_batch).float()
    one_hots = [F.one_hot(c.squeeze(-1), n).float() for c, n in zip([batch, *cats], [n_batch, *n_cats_per_cov]) if n > 1]
    covariates = torch.cat(one_hots, dim=1) if one_hots else torch.zeros(N_CELLS, 0)
    tensors = {k: v.to(device) for k, v in tensors.items()}
    return tensors, x.to(device), batch_one_hot.to(device), covariates.to(device)


def train_step(model, loss_fn):
    """One Adam step with scVI's optimizer settings, from a fixed RNG state (the latent draw); returns the loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=0.01, weight_decay=1e-6)
    torch.manual_seed(2)
    model.train()
    loss = loss_fn()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach()


def assert_matches_scvi_weights(model, reference):
    converted = EtudeSCVI.state_dict_from_scvi(reference.state_dict())
    state = model.state_dict()
    assert state.keys() == converted.keys()
    different = [k for k in state if not torch.allclose(state[k].float(), converted[k].float(), **TOLERANCE)]
    assert not different, f'tensors differ: {different}'


# ---------------------------------------------------------------------------
# Same model as scvi-tools 1.4.2's VAE: with scVI's weights, EtudeSCVI computes the same results
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_loads_scvi_weights(n_batch, n_cats_per_cov, n_layers):
    # Every weight of EtudeSCVI has a counterpart in scVI's VAE with the same shape (strict loading)
    reference, model = matched_pair(n_batch, n_cats_per_cov, n_layers)
    assert_matches_scvi_weights(model, reference)


@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_keeps_only_used_weights(n_batch, n_cats_per_cov, n_layers):
    # scVI's library encoder and its dispersion and dropout heads are unused with these settings; EtudeSCVI has none
    reference, model = matched_pair(n_batch, n_cats_per_cov, n_layers)
    assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in reference.parameters())


@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_posterior_matches_scvi(n_batch, n_cats_per_cov, n_layers):
    # scVI's latent representation is the posterior mean, in eval mode
    reference, model = matched_pair(n_batch, n_cats_per_cov, n_layers)
    tensors, x, _, _ = make_batch(n_batch, n_cats_per_cov)
    reference.eval(), model.eval()
    with torch.no_grad():
        expected = reference._regular_inference(x, tensors['batch'])['qz']
        posterior = model.posterior(x)
    assert torch.allclose(posterior.loc, expected.loc, **TOLERANCE)
    assert torch.allclose(posterior.scale, expected.scale, **TOLERANCE)


@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_loss_terms_match_scvi(n_batch, n_cats_per_cov, n_layers):
    # Per-cell reconstruction loss and KL divergence (which scVI's ELBO metrics and early stopping average), and the
    # loss; in eval mode the latent draw is the only random step, the same standard normal draw for both
    reference, model = matched_pair(n_batch, n_cats_per_cov, n_layers)
    tensors, x, batch_one_hot, covariates = make_batch(n_batch, n_cats_per_cov)
    kl_weight = torch.tensor(0.5)
    reference.eval(), model.eval()
    with torch.no_grad():
        torch.manual_seed(3)
        expected = reference(tensors, loss_kwargs={'kl_weight': kl_weight})[2]
        torch.manual_seed(3)
        terms = model.loss_terms(x, batch_one_hot, covariates, kl_weight)
    assert torch.allclose(terms['reconstruction'], expected.reconstruction_loss['reconstruction_loss'], **TOLERANCE)
    assert torch.allclose(terms['kl'], expected.kl_local['kl_divergence_z'], **TOLERANCE)
    assert torch.allclose(terms['loss'], expected.loss, **TOLERANCE)


def assert_training_step_matches_scvi(n_batch, n_cats_per_cov, n_layers, device):
    # Without dropout, a training step's only random step is the latent draw, so both models see the same numbers
    reference, model = matched_pair(n_batch, n_cats_per_cov, n_layers, dropout_rate=0.0)
    reference, model = reference.to(device), model.to(device)
    tensors, x, batch_one_hot, covariates = make_batch(n_batch, n_cats_per_cov, device)
    kl_weight = torch.tensor(5 / 400, device=device)  # scVI's KL weight in epoch 5 of its 400-epoch warmup

    reference_loss = train_step(reference, lambda: reference(tensors, loss_kwargs={'kl_weight': kl_weight})[2].loss)
    loss = train_step(model, lambda: model.loss_terms(x, batch_one_hot, covariates, kl_weight)['loss'])
    assert torch.allclose(loss, reference_loss, **TOLERANCE)
    assert_matches_scvi_weights(model, reference)


@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_training_step_matches_scvi(n_batch, n_cats_per_cov, n_layers):
    assert_training_step_matches_scvi(n_batch, n_cats_per_cov, n_layers, 'cpu')


@GPU
@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_training_step_matches_scvi_on_gpu(n_batch, n_cats_per_cov, n_layers):
    assert_training_step_matches_scvi(n_batch, n_cats_per_cov, n_layers, 'cuda')


# ---------------------------------------------------------------------------
# Same initialization as scVI: the log inverse dispersion starts from a standard normal draw
# ---------------------------------------------------------------------------
def test_dispersion_starts_standard_normal():
    torch.manual_seed(0)
    model = EtudeSCVI(2000, 50, **ARCH)
    log_theta = model.state_dict()[[k for k in model.state_dict() if 'dispersion' in k][0]]
    assert log_theta.shape == (2000, 50)
    assert abs(log_theta.mean().item()) < 0.02 and abs(log_theta.std().item() - 1) < 0.02


# ---------------------------------------------------------------------------
# Without scvi-tools: EtudeSCVI trains and does not import scvi-tools
# ---------------------------------------------------------------------------
def test_trains_a_few_steps():
    torch.manual_seed(0)
    model = EtudeSCVI(N_GENES, 3, (5,), n_layers=3, **ARCH)
    _, x, batch_one_hot, covariates = make_batch(3, (5,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=0.01, weight_decay=1e-6)
    losses = []
    for _ in range(20):
        loss = model.loss_terms(x, batch_one_hot, covariates, torch.tensor(0.0))['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses))) and losses[-1] < losses[0]


def test_does_not_import_scvi_tools():
    # A fresh interpreter, run from the repository root so it imports this checkout of piano
    code = 'import sys, piano.models; print("scvi" in sys.modules)'
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'False'
