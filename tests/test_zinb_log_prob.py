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

import pytest
import torch
from pyro.distributions import ZeroInflatedNegativeBinomial

from piano.models.base_models import Etude, EtudeMuTheta

N_CELLS = 32
N_GENES = 64
N_COVARIATE_DIMS = 3  # One categorical covariate with 3 one-hot columns
MODEL_CLASSES = {'ksi-psi': Etude, 'mu-theta': EtudeMuTheta}

# pyro stays the reference for the ZINB log-likelihood (SCRUM-46). The plain-torch helper copies pyro's ops in the same
# order, so it was bit-identical in eager mode (plan check job 16102321), but only a close result is required: these
# tolerances would also accept a mathematically equal formula such as logaddexp (gradient differences up to 3.4e-5)
RTOL = 1e-5
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Helpers: a small ZINB model and a batch of counts with a one-hot covariate, on CPU
# ---------------------------------------------------------------------------
def make_model(parameterization, adversarial, seed=0):
    torch.manual_seed(seed)  # Same weights for every call with the same seed
    model = MODEL_CLASSES[parameterization](
        input_size=N_GENES, n_hidden=16, n_layers=2, latent_size=4,
        n_total_covariate_dims=N_COVARIATE_DIMS, n_categorical_covariate_dims=N_COVARIATE_DIMS,
        distribution='zinb', adversarial=adversarial,
    )
    return model.train()


def make_batch(seed=1):
    # NB counts with extra zeros, as ZINB is for, followed by the one-hot covariate columns
    generator = torch.Generator().manual_seed(seed)
    counts = torch.poisson(torch.full((N_CELLS, N_GENES), 2.0), generator=generator)
    counts[torch.rand(N_CELLS, N_GENES, generator=generator) < 0.4] = 0
    covariates = torch.eye(N_COVARIATE_DIMS)[torch.randint(0, N_COVARIATE_DIMS, (N_CELLS,), generator=generator)]
    return torch.cat([counts, covariates], dim=1)


def pyro_log_prob(total_count, logits, gate_logits, x):
    # The reference: pyro's ZINB, as the model's loss built it before SCRUM-46
    return ZeroInflatedNegativeBinomial(
        total_count=total_count, logits=logits, gate_logits=gate_logits, validate_args=False,
    ).log_prob(x)


def pyro_nll_loss(model):
    # The model's ZINB loss with pyro's distribution, to bind to one model instance in place of its own _nll_loss
    if isinstance(model, EtudeMuTheta):
        def nll_loss(mu, logtheta, x, zi_dropout_logits=None):
            # Same clamps and parameters as EtudeMuTheta._nll_loss_zinb
            logtheta = torch.clamp(logtheta, model.min_logit, model.max_logit)
            theta = torch.clamp(torch.exp(logtheta), min=model.min_clip, max=model.max_ksi_clip)
            logits = torch.log(mu) - logtheta
            return -pyro_log_prob(theta, logits, zi_dropout_logits, x).sum()
    else:
        def nll_loss(nb_ksi, nb_psi, x, zi_dropout_logits=None):
            return -pyro_log_prob(nb_ksi, nb_psi, zi_dropout_logits, x).sum()
    return nll_loss


def log_prob_and_gradients(log_prob, total_count, logits, gate_logits, x):
    # Elementwise log-likelihood and its gradients with respect to the three distribution parameters
    params = [p.clone().requires_grad_(True) for p in (total_count, logits, gate_logits)]
    values = log_prob(*params, x)
    gradients = torch.autograd.grad(values.sum(), params)
    return values.detach(), gradients


def prefixed(name):
    # An assert_close message that keeps the details and says which tensor differs
    return lambda message: f'{name}: {message}'


def describe_break(break_reason):
    # A graph break's first reason line and the code it comes from: the last frame outside installed packages (piano or
    # this test), as the last frame of all is often deep inside torch
    reason = str(break_reason.reason).splitlines()[0]
    frames = [frame for frame in break_reason.user_stack if 'site-packages' not in frame.filename]
    if not frames:
        return reason
    return f'{reason} at {frames[-1].filename}:{frames[-1].lineno}'


def extreme_inputs():
    # Every combination of total_count 1e-8 to 1e8 (the model's clamps), logits and gate logits at -20, 0 and 20 (the
    # clamps), and counts 0, 1 and 65535 (the largest uint16 count)
    grid = torch.cartesian_prod(
        torch.tensor([1e-8, 1e-3, 1.0, 1e4, 1e8]),
        torch.tensor([-20.0, 0.0, 20.0]),
        torch.tensor([-20.0, 0.0, 20.0]),
        torch.tensor([0.0, 1.0, 65535.0]),
    )
    return grid.unbind(dim=1)


def typical_inputs(seed=2):
    # Random parameters in the range a trained model produces, and counts with many zeros
    generator = torch.Generator().manual_seed(seed)
    shape = (N_CELLS, N_GENES)
    total_count = torch.exp(torch.randn(shape, generator=generator) * 2)
    logits = torch.randn(shape, generator=generator) * 3
    gate_logits = torch.randn(shape, generator=generator) * 3
    x = torch.poisson(torch.full(shape, 1.0), generator=generator)
    return total_count, logits, gate_logits, x


# ---------------------------------------------------------------------------
# The compiled ZINB forward has no graph breaks from the loss (SCRUM-46, review item 39)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('adversarial', [True, False], ids=['adversarial', 'no_adversarial'])
@pytest.mark.parametrize('parameterization', ['ksi-psi', 'mu-theta'])
def test_zinb_forward_has_no_graph_breaks(parameterization, adversarial):
    # Building pyro's distribution on every forward broke the graph 4 times (Etude) or 5 times (EtudeMuTheta); NB has 0
    model = make_model(parameterization, adversarial)
    torch._dynamo.reset()
    try:
        explanation = torch._dynamo.explain(model.forward)(make_batch())
    finally:
        torch._dynamo.reset()  # Keep dynamo's caches and counters from leaking into later tests
    reasons = [describe_break(reason) for reason in explanation.break_reasons]
    assert explanation.graph_break_count == 0, reasons


# ---------------------------------------------------------------------------
# The plain-torch ZINB log-likelihood matches pyro's ZeroInflatedNegativeBinomial.log_prob
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('inputs', [typical_inputs, extreme_inputs], ids=['typical', 'extreme'])
def test_zinb_log_prob_matches_pyro(inputs):
    from piano.models.base_models import _zinb_log_prob  # Inside the test, so that only this test fails without it

    total_count, logits, gate_logits, x = inputs()
    expected, expected_gradients = log_prob_and_gradients(pyro_log_prob, total_count, logits, gate_logits, x)
    actual, actual_gradients = log_prob_and_gradients(_zinb_log_prob, total_count, logits, gate_logits, x)

    assert torch.isfinite(actual).all()
    assert all(torch.isfinite(gradient).all() for gradient in actual_gradients)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    for name, actual_gradient, expected_gradient in zip(
            ('total_count', 'logits', 'gate_logits'), actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=RTOL, atol=ATOL, msg=prefixed(name))


# ---------------------------------------------------------------------------
# Coverage: the models' ZINB loss and parameter gradients match a pyro-built loss (passes on master, where it is pyro)
# ---------------------------------------------------------------------------
def loss_and_gradients(model, x_aug):
    torch.manual_seed(5)  # Same posterior draw and dropout masks for both losses
    losses = model.training_step(x_aug, 1.0, 1.0)
    losses['total'].backward()
    gradients = {name: param.grad for (name, param) in model.named_parameters() if param.grad is not None}
    return losses['nll'].detach(), gradients


@pytest.mark.parametrize('adversarial', [True, False], ids=['adversarial', 'no_adversarial'])
@pytest.mark.parametrize('parameterization', ['ksi-psi', 'mu-theta'])
def test_zinb_nll_and_gradients_match_pyro(parameterization, adversarial):
    x_aug = make_batch()
    model = make_model(parameterization, adversarial)
    reference = make_model(parameterization, adversarial)  # Same weights
    reference._nll_loss = pyro_nll_loss(reference)  # Per instance, so the class is unchanged

    nll, gradients = loss_and_gradients(model, x_aug)
    expected_nll, expected_gradients = loss_and_gradients(reference, x_aug)

    torch.testing.assert_close(nll, expected_nll, rtol=RTOL, atol=ATOL)
    assert gradients.keys() == expected_gradients.keys()
    for name, gradient in gradients.items():
        torch.testing.assert_close(gradient, expected_gradients[name], rtol=RTOL, atol=ATOL, msg=prefixed(name))
