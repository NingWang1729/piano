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
from torch.distributions import Normal, kl_divergence

from piano.models import EtudeSCVI

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CELLS = 64
N_GENES = 30
ARCH = {'n_hidden': 16, 'latent_size': 4}

# Float tolerance: the same model computed with a different order of operations
TOLERANCE = {'rtol': 1e-5, 'atol': 1e-5}

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')

# (n_batch, n_cats_per_cov, n_layers): one hidden layer (scVI's default), three (a deeper network), an extra categorical
# covariate, and a single batch or a single-category covariate (which scVI leaves out of the decoder, as they carry no
# information)
CONFIGS = [(3, (), 1), (3, (), 3), (3, (5,), 3), (1, (), 2), (3, (1, 5), 2)]

# Likelihoods, each with the Poisson rate of its test counts: ZINB's sparse counts have many zeros and many positive
# counts, so both of its cases (zero and positive count) are exercised
DISTRIBUTIONS = ['nb', 'zinb']
COUNT_RATE = {'nb': 3.0, 'zinb': 0.7}


# ---------------------------------------------------------------------------
# Helpers: scVI's own VAE and EtudeSCVI with the same weights, and one minibatch in both input formats
#
# scvi-tools is only a reference here (PIANO's optional scvi-tools extra); tests that need it skip without it
# ---------------------------------------------------------------------------
def scvi_vae(n_batch, n_cats_per_cov, n_layers, dropout_rate=0.1, distribution='nb'):
    vae_module = pytest.importorskip('scvi.module')
    return vae_module.VAE(
        N_GENES, n_batch=n_batch, n_labels=1, n_cats_per_cov=list(n_cats_per_cov) or None, n_layers=n_layers,
        n_hidden=ARCH['n_hidden'], n_latent=ARCH['latent_size'], dropout_rate=dropout_rate, gene_likelihood=distribution,
        dispersion='gene-batch',
    )


def matched_pair(n_batch, n_cats_per_cov, n_layers, dropout_rate=0.1, distribution='nb'):
    """scVI's VAE and an EtudeSCVI that loaded its weights."""
    torch.manual_seed(0)
    reference = scvi_vae(n_batch, n_cats_per_cov, n_layers, dropout_rate, distribution)
    model = EtudeSCVI(
        N_GENES, n_batch, n_cats_per_cov, n_layers=n_layers, dropout_rate=dropout_rate, distribution=distribution, **ARCH,
    )
    model.load_scvi_state_dict(reference.state_dict())
    return reference, model


def make_batch(n_batch, n_cats_per_cov, device='cpu', seed=1, rate=3.0):
    """scVI's tensors dict and EtudeSCVI's inputs (counts, batch one-hot, decoder covariates) for the same cells."""
    g = torch.Generator().manual_seed(seed)
    x = torch.poisson(torch.full((N_CELLS, N_GENES), rate), generator=g)
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
    converted = EtudeSCVI.state_dict_from_scvi(reference.state_dict(), model.distribution)
    state = model.state_dict()
    assert state.keys() == converted.keys()
    different = [k for k in state if not torch.allclose(state[k].float(), converted[k].float(), **TOLERANCE)]
    assert not different, f'tensors differ: {different}'


# ---------------------------------------------------------------------------
# Same model as scvi-tools 1.4.2's VAE: with scVI's weights, EtudeSCVI computes the same results
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('distribution', DISTRIBUTIONS)
@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_loads_scvi_weights(n_batch, n_cats_per_cov, n_layers, distribution):
    # Every weight of EtudeSCVI has a counterpart in scVI's VAE with the same shape (strict loading)
    reference, model = matched_pair(n_batch, n_cats_per_cov, n_layers, distribution=distribution)
    assert_matches_scvi_weights(model, reference)


@pytest.mark.parametrize('distribution', DISTRIBUTIONS)
@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_keeps_only_used_weights(n_batch, n_cats_per_cov, n_layers, distribution):
    # scVI's library encoder and its dispersion head are unused with these settings, and its dropout head with NB;
    # EtudeSCVI has none of them
    reference, model = matched_pair(n_batch, n_cats_per_cov, n_layers, distribution=distribution)
    assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in reference.parameters())


@pytest.mark.parametrize('distribution', DISTRIBUTIONS)
@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_posterior_matches_scvi(n_batch, n_cats_per_cov, n_layers, distribution):
    # scVI's latent representation is the posterior mean, in eval mode
    reference, model = matched_pair(n_batch, n_cats_per_cov, n_layers, distribution=distribution)
    tensors, x, _, _ = make_batch(n_batch, n_cats_per_cov, rate=COUNT_RATE[distribution])
    reference.eval(), model.eval()
    with torch.no_grad():
        expected = reference._regular_inference(x, tensors['batch'])['qz']
        posterior = model.posterior(x)
    assert torch.allclose(posterior.loc, expected.loc, **TOLERANCE)
    assert torch.allclose(posterior.scale, expected.scale, **TOLERANCE)


@pytest.mark.parametrize('distribution', DISTRIBUTIONS)
@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_loss_terms_match_scvi(n_batch, n_cats_per_cov, n_layers, distribution):
    # Per-cell reconstruction loss and KL divergence (which scVI's ELBO metrics and early stopping average), and the
    # loss; in eval mode the latent draw is the only random step, the same standard normal draw for both
    reference, model = matched_pair(n_batch, n_cats_per_cov, n_layers, distribution=distribution)
    tensors, x, batch_one_hot, covariates = make_batch(n_batch, n_cats_per_cov, rate=COUNT_RATE[distribution])
    if distribution == 'zinb':
        assert 0.3 < (x == 0).float().mean() < 0.7  # Both ZINB cases: zero and positive counts
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


def assert_training_step_matches_scvi(n_batch, n_cats_per_cov, n_layers, device, distribution):
    # Without dropout, a training step's only random step is the latent draw, so both models see the same numbers
    reference, model = matched_pair(n_batch, n_cats_per_cov, n_layers, dropout_rate=0.0, distribution=distribution)
    reference, model = reference.to(device), model.to(device)
    tensors, x, batch_one_hot, covariates = make_batch(n_batch, n_cats_per_cov, device, rate=COUNT_RATE[distribution])
    kl_weight = torch.tensor(5 / 400, device=device)  # scVI's KL weight in epoch 5 of its 400-epoch warmup

    reference_loss = train_step(reference, lambda: reference(tensors, loss_kwargs={'kl_weight': kl_weight})[2].loss)
    loss = train_step(model, lambda: model.loss_terms(x, batch_one_hot, covariates, kl_weight)['loss'])
    assert torch.allclose(loss, reference_loss, **TOLERANCE)
    assert_matches_scvi_weights(model, reference)


@pytest.mark.parametrize('distribution', DISTRIBUTIONS)
@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_training_step_matches_scvi(n_batch, n_cats_per_cov, n_layers, distribution):
    assert_training_step_matches_scvi(n_batch, n_cats_per_cov, n_layers, 'cpu', distribution)


@GPU
@pytest.mark.parametrize('distribution', DISTRIBUTIONS)
@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_training_step_matches_scvi_on_gpu(n_batch, n_cats_per_cov, n_layers, distribution):
    assert_training_step_matches_scvi(n_batch, n_cats_per_cov, n_layers, 'cuda', distribution)


# ---------------------------------------------------------------------------
# Likelihood: NB (default) or ZINB, which adds only a dropout head
# ---------------------------------------------------------------------------
def test_invalid_distribution_raises():
    # Another likelihood could exist later, so NotImplementedError, as the Composer raises; also when loading weights
    with pytest.raises(NotImplementedError, match='distribution'):
        EtudeSCVI(N_GENES, 3, distribution='poisson', **ARCH)
    with pytest.raises(NotImplementedError, match='distribution'):
        EtudeSCVI.state_dict_from_scvi({}, distribution='ZINB')


def test_zinb_adds_only_a_dropout_head():
    # NB is the default and has no dropout head; ZINB adds one linear layer from the decoder's hidden units to the genes,
    # named decoder_dropouts as Etude's ZINB head
    nb_state = EtudeSCVI(N_GENES, 3, (5,), n_layers=2, **ARCH).state_dict()
    zinb_state = EtudeSCVI(N_GENES, 3, (5,), n_layers=2, distribution='zinb', **ARCH).state_dict()
    assert nb_state.keys() == EtudeSCVI(N_GENES, 3, (5,), n_layers=2, distribution='nb', **ARCH).state_dict().keys()
    added = {k: tuple(v.shape) for k, v in zinb_state.items() if k not in nb_state}
    assert set(nb_state) < set(zinb_state)
    assert added == {'decoder_dropouts.weight': (N_GENES, ARCH['n_hidden']), 'decoder_dropouts.bias': (N_GENES,)}


def test_scvi_zinb_dropout_head_converts_to_decoder_dropouts():
    # scvi-tools' ZINB dropout head becomes decoder_dropouts, and the converted state_dict loads strictly
    reference = scvi_vae(3, (5,), 2, distribution='zinb')
    converted = EtudeSCVI.state_dict_from_scvi(reference.state_dict(), distribution='zinb')
    assert torch.equal(converted['decoder_dropouts.weight'], reference.state_dict()['decoder.px_dropout_decoder.weight'])
    model = EtudeSCVI(N_GENES, 3, (5,), n_layers=2, distribution='zinb', **ARCH)
    model.load_state_dict(converted, strict=True)


def test_zinb_log_likelihood_matches_formula():
    # log(pi + (1 - pi) * NB(0)) for zero counts and log(1 - pi) + log NB(x) for positive counts, pi = sigmoid(logit),
    # computed directly in float64 (log NB from the model's NB formula, which the scvi-tools comparisons check); also
    # for large logits of both signs
    model = EtudeSCVI(N_GENES, 1, distribution='zinb', **ARCH)
    g = torch.Generator().manual_seed(0)
    x = torch.poisson(torch.full((N_CELLS, N_GENES), 0.7), generator=g)
    mu = torch.exp(torch.randn(N_CELLS, N_GENES, generator=g))
    theta = torch.exp(torch.randn(N_CELLS, N_GENES, generator=g))
    logits = torch.randn(N_CELLS, N_GENES, generator=g) * 3
    logits[0, :] = 30.0
    logits[1, :] = -30.0

    # 1 - pi as sigmoid(-logit): computed as 1 - sigmoid(logit), it would lose its digits to cancellation for large logits
    x64, mu64, theta64 = x.double(), mu.double(), theta.double()
    pi, one_minus_pi = torch.sigmoid(logits.double()), torch.sigmoid(-logits.double())
    nb = model._nb_log_likelihood(x64, mu64, theta64)
    nb_zero = theta64 * (torch.log(theta64 + 1e-8) - torch.log(theta64 + mu64 + 1e-8))
    expected = torch.where(x64 == 0, torch.log(pi + one_minus_pi * torch.exp(nb_zero)), torch.log(one_minus_pi) + nb)
    loglik = model._zinb_log_likelihood(x, mu, theta, logits)
    assert torch.allclose(loglik.double(), expected, **TOLERANCE)


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
# PIANO's augmented matrix: [counts | one-hot batch | one-hot of each other categorical covariate], as Composer builds it
# ---------------------------------------------------------------------------
def augmented_matrix(n_batch, n_cats_per_cov, seed=1):
    """The augmented matrix of make_batch's cells, with a column for every category (also of single-category covariates)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.poisson(torch.full((N_CELLS, N_GENES), 3.0), generator=g)
    batch = torch.randint(0, n_batch, (N_CELLS, 1), generator=g)
    cats = [torch.randint(0, n, (N_CELLS, 1), generator=g) for n in n_cats_per_cov]
    one_hots = [F.one_hot(c.squeeze(-1), n).float() for c, n in zip([batch, *cats], [n_batch, *n_cats_per_cov])]
    return torch.cat([x, *one_hots], dim=1)


@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_training_step_reads_augmented_matrix(n_batch, n_cats_per_cov, n_layers):
    # The same loss as loss_terms on the separate inputs, with per-batch sums of the per-cell terms for epoch averages
    torch.manual_seed(0)
    model = EtudeSCVI(N_GENES, n_batch, n_cats_per_cov, n_layers=n_layers, **ARCH)
    _, x, batch_one_hot, covariates = make_batch(n_batch, n_cats_per_cov)
    kl_weight = torch.tensor(0.5)
    model.eval()
    with torch.no_grad():
        torch.manual_seed(3)
        expected = model.loss_terms(x, batch_one_hot, covariates, kl_weight)
        torch.manual_seed(3)
        losses = model.training_step(augmented_matrix(n_batch, n_cats_per_cov), kl_weight, adv_weight=None)
    assert torch.equal(losses['loss'], expected['loss'])
    assert torch.equal(losses['reconstruction'], expected['reconstruction'].sum())
    assert torch.equal(losses['kl'], expected['kl'].sum())


@pytest.mark.parametrize('n_batch, n_cats_per_cov, n_layers', CONFIGS)
def test_latent_representation_is_posterior_mean(n_batch, n_cats_per_cov, n_layers):
    torch.manual_seed(0)
    model = EtudeSCVI(N_GENES, n_batch, n_cats_per_cov, n_layers=n_layers, **ARCH)
    model.to(model.device)  # As the Composer does: batches go to model.device, the GPU when one is available
    x_aug = augmented_matrix(n_batch, n_cats_per_cov)
    batches = list(x_aug.split(16))  # Any iterable of augmented batches, as Composer's DataLoader
    model.train()
    latent = model.get_latent_representation(batches)
    assert model.training  # Restored after inference in eval mode
    model.eval()
    with torch.no_grad():
        expected = model.posterior(x_aug[:, :N_GENES].to(model.device)).loc.cpu()
    assert latent.shape == (N_CELLS, ARCH['latent_size'])
    assert torch.allclose(torch.from_numpy(latent), expected, **TOLERANCE)  # Minibatches vs all cells at once

    # mc_samples averages that many posterior draws, which differ from the mean
    sampled = model.get_latent_representation(batches, mc_samples=3)
    assert sampled.shape == latent.shape and not (sampled == latent).all()


# ---------------------------------------------------------------------------
# Fast compiled training: the KL in closed form, so the training step compiles into one graph
# ---------------------------------------------------------------------------
def test_kl_matches_torch_distributions():
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(N_CELLS, ARCH['latent_size'], generator=g) * 2
    sigma = torch.exp(torch.randn(N_CELLS, ARCH['latent_size'], generator=g))
    expected = kl_divergence(Normal(mu, sigma), Normal(0.0, 1.0)).sum(dim=-1)
    kl = EtudeSCVI._kl_to_standard_normal(Normal(mu, sigma))
    assert kl.shape == (N_CELLS,)
    assert torch.allclose(kl, expected, **TOLERANCE)


@pytest.mark.parametrize('distribution', DISTRIBUTIONS)
def test_training_step_compiles_into_one_graph(distribution):
    # A fresh interpreter, as on the first training step: torch.distributions memoizes its KL lookup after an eager call,
    # which would hide a graph break from that lookup in this process
    code = '\n'.join([
        'import torch',
        'import torch.nn.functional as F',
        'from piano.models import EtudeSCVI',
        'torch.manual_seed(0)',
        f'model = EtudeSCVI(30, 3, n_hidden=16, latent_size=4, distribution={distribution!r})',
        'x = torch.poisson(torch.full((64, 30), 3.0))',
        'x_aug = torch.cat([x, F.one_hot(torch.randint(0, 3, (64,)), 3).float()], dim=1)',
        'explanation = torch._dynamo.explain(model.training_step)(x_aug, torch.tensor(0.5))',
        'print(explanation.graph_count, explanation.graph_break_count)',
    ])
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['1', '0']  # One graph, no graph breaks


# ---------------------------------------------------------------------------
# Without scvi-tools: EtudeSCVI trains and does not import scvi-tools
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('distribution', DISTRIBUTIONS)
def test_trains_a_few_steps(distribution):
    torch.manual_seed(0)
    model = EtudeSCVI(N_GENES, 3, (5,), n_layers=3, distribution=distribution, **ARCH)
    _, x, batch_one_hot, covariates = make_batch(3, (5,), rate=COUNT_RATE[distribution])
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
