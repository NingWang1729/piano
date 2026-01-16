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

from typing import Literal

import torch
import torch.nn.functional as F
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from torch import nn
from torch.distributions import NegativeBinomial, Normal
from torch.distributions.kl import _kl_normal_normal


# Gradient reversal
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class Etude(nn.Module):
    def __init__(
        self,

        # Model architecture
        input_size: int = 4096,  # Must be Python int
        n_hidden: int = 256,
        n_layers: int = 3,
        latent_size: int = 32,
        cov_size: int = 0,

        # Model hyperparameters
        dropout_rate: float = 0.1,
        batchnorm_eps: float = 1e-5,       # Torch default is 1e-5
        batchnorm_momentum: float = 1e-1,  # Torch default is 1e-1
        epsilon: float = 1e-5,             # Torch default is 1e-5

        # Padding (only used for padded child classes)
        padding_size: int = 0,  # Must be Python int

        # Save batch keys (only used for scVI models)
        n_batch_keys: int = 1,  # Number of batches in batch key
    ):
        super().__init__()

        # Save architecture
        self.input_size = int(input_size)  # Must be Python int
        self.n_hidden = int(n_hidden)
        self.n_layers = int(n_layers)
        self.latent_size = int(latent_size)
        self.cov_size = int(cov_size)

        # Save hyperparameters
        self.dropout_rate = dropout_rate
        self.bn_eps = batchnorm_eps
        self.bn_moment = batchnorm_momentum
        self.epsilon = epsilon

        # Save padding (used only for padded child classes)
        self.padding_size = padding_size
        # Save batch keys (only used for scVI models)
        self.n_batch_keys = n_batch_keys

        # Training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Encoder layers
        layers = []
        layers.append(nn.Linear(self.input_size, self.n_hidden))
        layers.append(nn.BatchNorm1d(self.n_hidden, eps=self.bn_eps, momentum=self.bn_moment))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(self.n_hidden, self.n_hidden))
            layers.append(nn.BatchNorm1d(self.n_hidden, eps=self.bn_eps, momentum=self.bn_moment))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_rate))
        self.encoder_layers = nn.Sequential(*layers)
        self.encoder_mean = nn.Linear(self.n_hidden, self.latent_size)
        self.encoder_log_var = nn.Linear(self.n_hidden, self.latent_size)

        # Adversarial batch classifier
        self.batch_classifier = nn.Sequential(
            nn.Linear(self.latent_size, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.cov_size),
        )

        # Decoder layers
        layers = []
        layers.append(nn.Linear(self.latent_size + self.cov_size, self.n_hidden))
        layers.append(nn.BatchNorm1d(self.n_hidden, eps=self.bn_eps, momentum=self.bn_moment))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(self.n_hidden, self.n_hidden))
            layers.append(nn.BatchNorm1d(self.n_hidden, eps=self.bn_eps, momentum=self.bn_moment))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_rate))
        self.decoder_layers = nn.Sequential(*layers)
        self.decoder_recon = nn.Linear(self.n_hidden, self.input_size)
        self.decoder_recon_act = nn.Softmax(dim=-1)

        # Initialize GLM weights
        self.b_mu = nn.Parameter(torch.ones(1, self.input_size))  # Shape (1, G)
        self.w_mu_gene = nn.Parameter(torch.ones(self.input_size))  # Shape (G)
        self.w_mu_lib = nn.Parameter(torch.ones(1, self.input_size))  # Shape (1, G)
        self.w_mu_cov = nn.Parameter(torch.zeros(self.cov_size, self.input_size))  # Shape (B, G)
        self.b_psi = nn.Parameter(torch.ones(1, self.input_size))  # Shape (1, G)
        self.w_psi = nn.Parameter(torch.zeros(self.cov_size, self.input_size))  # Shape (B, G)

        # Numerical stability
        self.max_logit = 20
        self.min_logit = -20
        self.max_mu_clip = 2e4
        self.max_ksi_clip = 1e8
        self.min_clip = 1e-8
        self.min_library_size = 2e2

    def forward(self, x_aug, adv_lambda=0.0):
        # Extract gene data and covariates from [X_genes; X_covariates]
        x = x_aug[:, :self.input_size]
        covariates_matrix = x_aug[:, self.input_size:]
        library = torch.sum(x, dim=1, keepdim=True)  # Shape (N, 1)

        # Log1p transformation for stability
        x_encoded = torch.log1p(x)  # Shape (N, G)

        # Run inference
        x_encoded = self.encoder_layers(x_encoded)  # Shape (N, H)

        # Latent posterior distribution q(z | x)
        posterior_mu = self.encoder_mean(x_encoded)
        posterior_log_var = self.encoder_log_var(x_encoded)
        posterior_sigma = torch.clamp(
            torch.exp(0.5 * posterior_log_var), min=self.epsilon
        )  # Shape (N, Z)

        # Construct posterior distribution
        posterior_dist = Normal(posterior_mu, posterior_sigma)

        # Reparameterization trick
        posterior_latent = posterior_dist.rsample()  # Shape (N, Z)

        # Adversarial batch prediction
        z_adv = grad_reverse(posterior_latent, adv_lambda)
        batch_logits = self.batch_classifier(z_adv)
        adv_loss = F.binary_cross_entropy_with_logits(
            batch_logits,
            covariates_matrix,
            reduction='sum',
        )

        # Calculate KL divergence penalty
        prior_dist = Normal(torch.zeros_like(posterior_mu), torch.ones_like(posterior_mu))
        kld_loss = _kl_normal_normal(posterior_dist, prior_dist).sum()  # Shape (N, Z)

        # Run generative model
        x_decoded = torch.cat([posterior_latent, covariates_matrix], dim=1)  # Shape (N, Z + B)
        x_decoded = self.decoder_layers(x_decoded)  # Shape (N, H)
        x_bar = self.decoder_recon(x_decoded)  # Shape (N, G)
        x_bar = self.decoder_recon_act(x_bar)  # Shape (N, G)

        # Parameterize (ZI)NB
        nb_mu = torch.clamp(
            torch.exp(
                torch.clamp(
                    (
                        torch.ones_like(library) @ self.b_mu  # Shape (N, 1) @ (1, G)
                        + torch.log(torch.clamp(x_bar, min=self.min_clip)) * self.w_mu_gene  # Shape (N, G) * (G)
                        + torch.log(torch.clamp(library, min=self.min_library_size)) @ self.w_mu_lib  # Shape (N, 1) @ (1, G)
                        + covariates_matrix @ self.w_mu_cov  # Shape (N, B) @ (B, G)
                    ),
                    min=self.min_logit,  # Numerical stability
                    max=self.max_logit,  # Numerical stability
                )
            ),
            min=self.min_clip,  # Numerical stability
            max=self.max_mu_clip,  # Numerical stability
        )  # Shape (N, G)
        nb_psi = torch.clamp(
            (
                torch.ones_like(library) @ self.b_psi  # Shape (N, 1) @ (1, G)
                + covariates_matrix @ self.w_psi  # Shape (N, B) @ (B, G)
            ),
            min=self.min_logit,  # Numerical stability
            max=self.max_logit,  # Numerical stability
        )  # Shape (N, G)
        nb_ksi = torch.clamp(
            nb_mu * torch.exp(-nb_psi),
            min=self.min_clip,  # Numerical stability
            max=self.max_ksi_clip,  # Numerical stability
        )  # Shape (N, G)

        # Calculate NLL
        nll_loss = -NegativeBinomial(
            total_count=nb_ksi,  # Rate/overdispersion
            logits=nb_psi,  # Log-odds
            validate_args=False,
        ).log_prob(x).sum()

        # Return latent space
        return kld_loss, nll_loss, adv_loss

    def training_step(self, batch, kld_weight, adv_lambda):
        kld_loss, nll_loss, adv_loss = self.forward(batch, adv_lambda=adv_lambda)

        elbo_loss = (nll_loss + kld_loss * kld_weight) / (1 + kld_weight) + adv_loss

        return elbo_loss, nll_loss, kld_loss, adv_loss

    def get_batch_latent_representation(self, x_aug, mc_samples=0):
        # Run inference
        x = x_aug[:, :self.input_size]
        x_encoded = torch.log1p(x)  # Shape (N, G)
        x_encoded = self.encoder_layers(x_encoded)  # Shape (N, H)

        # Posterior distribution q(z | x)
        posterior_mu = self.encoder_mean(x_encoded)
        posterior_log_var = self.encoder_log_var(x_encoded)
        posterior_sigma = torch.clamp(
            torch.exp(0.5 * posterior_log_var), min=self.epsilon
        )  # Shape (N, Z)
        posterior_dist = Normal(posterior_mu, posterior_sigma)

        # Sample latent space representations
        if mc_samples > 0:
            posterior_latent_list = posterior_dist.sample([mc_samples])  # Shape (MC, N, Z)
            posterior_latent = torch.mean(posterior_latent_list, dim=0)  # Shape (N, Z)
        else:
            posterior_latent = posterior_dist.sample()  # Shape (N, Z)

        # Return latent space
        return posterior_latent

    def get_latent_representation(self, dataloader, mc_samples=0):
        previously_training = self.training
        self.eval()

        # Sample latent space representations
        latent_space_representations = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device=self.device, non_blocking=True) # For non-GPU memory modes
                posterior_latent = self.get_batch_latent_representation(
                    batch,
                    mc_samples=mc_samples,
                )
                latent_space_representations.append(posterior_latent)
        latent_space_representations = torch.cat(latent_space_representations, dim=0)

        if previously_training:
            self.train()

        return latent_space_representations

    def get_batch_counterfactuals(self, x_aug, covariates_matrix=None):
        # Extract gene data and covariates from [X_genes; X_covariates]
        x = x_aug[:, :self.input_size]
        if covariates_matrix is None:
            covariates_matrix = x_aug[:, self.input_size:]
        library = torch.sum(x, dim=1, keepdim=True)  # Shape (N, 1)
        x_encoded = torch.log1p(x)  # Shape (N, G)
        x_encoded = self.encoder_layers(x_encoded)  # Shape (N, H)

        # Latent posterior distribution q(z | x)
        posterior_mu = self.encoder_mean(x_encoded)
        posterior_log_var = self.encoder_log_var(x_encoded)
        posterior_sigma = torch.clamp(
            torch.exp(0.5 * posterior_log_var), min=self.epsilon
        )  # Shape (N, Z)
        posterior_dist = Normal(posterior_mu, posterior_sigma)
        posterior_latent = posterior_dist.sample()  # Shape (N, Z)

        # Run generative model
        x_decoded = torch.cat([posterior_latent, covariates_matrix], dim=1)  # Shape (N, Z + B)
        x_decoded = self.decoder_layers(x_decoded)  # Shape (N, H)
        x_bar = self.decoder_recon(x_decoded)  # Shape (N, G)
        x_bar = self.decoder_recon_act(x_bar)  # Shape (N, G)

        # Parameterize (ZI)NB
        nb_mu = torch.clamp(
            torch.exp(
                torch.clamp(
                    (
                        torch.ones_like(library) @ self.b_mu  # Shape (N, 1) @ (1, G)
                        + torch.log(torch.clamp(x_bar, min=self.min_clip)) * self.w_mu_gene  # Shape (N, G) * (G)
                        + torch.log(torch.clamp(library, min=self.min_library_size)) @ self.w_mu_lib  # Shape (N, 1) @ (1, G)
                        + covariates_matrix @ self.w_mu_cov  # Shape (N, B) @ (B, G)
                    ),
                    min=self.min_logit,  # Numerical stability
                    max=self.max_logit,  # Numerical stability
                )
            ),
            min=self.min_clip,  # Numerical stability
            max=self.max_mu_clip,  # Numerical stability
        )  # Shape (N, G)

        return nb_mu

    def get_counterfactuals(self, dataloader, covariates=None):
        previously_training = self.training
        self.eval()

        # Sample latent space representations
        counterfactuals = []
        with torch.no_grad():
            if covariates is not None:
                covariates = torch.as_tensor(
                    covariates,
                    dtype=torch.float32,
                    device=self.device,
                )
            for batch in dataloader:
                batch = batch.to(device=self.device, non_blocking=True) # For non-GPU memory modes
                if covariates is not None:
                    covariates_matrix = covariates.unsqueeze(0).expand(batch.shape[0], -1)
                else:
                    covariates_matrix = None
                counterfactual = self.get_batch_counterfactuals(batch, covariates_matrix=covariates_matrix)
                counterfactuals.append(counterfactual)
        counterfactuals = torch.cat(counterfactuals, dim=0)

        if previously_training:
            self.train()

        return counterfactuals

class ZinbEtude(Etude):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder_dropouts = nn.Linear(self.n_hidden, self.input_size)

    def forward(self, x_aug):
        # Extract gene data and covariates from [X_genes; X_covariates]
        x = x_aug[:, :self.input_size]
        covariates_matrix = x_aug[:, self.input_size:]
        library = torch.sum(x, dim=1, keepdim=True)  # Shape (N, 1)

        # Log1p transformation for stability
        x_encoded = torch.log1p(x)  # Shape (N, G)

        # Run inference
        x_encoded = self.encoder_layers(x_encoded)  # Shape (N, H)

        # Latent posterior distribution q(z | x)
        posterior_mu = self.encoder_mean(x_encoded)
        posterior_log_var = self.encoder_log_var(x_encoded)
        posterior_sigma = torch.clamp(
            torch.exp(0.5 * posterior_log_var), min=self.epsilon
        )  # Shape (N, Z)

        # Construct posterior distribution
        posterior_dist = Normal(posterior_mu, posterior_sigma)

        # Reparameterization trick
        posterior_latent = posterior_dist.rsample()  # Shape (N, Z)

        # Calculate KL divergence penalty
        prior_dist = Normal(torch.zeros_like(posterior_mu), torch.ones_like(posterior_mu))
        kld_loss = _kl_normal_normal(posterior_dist, prior_dist).sum()  # Shape (N, Z)

        # Run generative model
        x_decoded = torch.cat([posterior_latent, covariates_matrix], dim=1)  # Shape (N, Z + B)
        x_decoded = self.decoder_layers(x_decoded)  # Shape (N, H)
        x_bar = self.decoder_recon(x_decoded)  # Shape (N, G)
        x_bar = self.decoder_recon_act(x_bar)  # Shape (N, G)

        # Parameterize (ZI)NB
        nb_mu = torch.clamp(
            torch.exp(
                torch.clamp(
                    (
                        torch.ones_like(library) @ self.b_mu  # Shape (N, 1) @ (1, G)
                        + torch.log(torch.clamp(x_bar, min=self.min_clip)) * self.w_mu_gene  # Shape (N, G) * (G)
                        + torch.log(torch.clamp(library, min=self.min_library_size)) @ self.w_mu_lib  # Shape (N, 1) @ (1, G)
                        + covariates_matrix @ self.w_mu_cov  # Shape (N, B) @ (B, G)
                    ),
                    min=self.min_logit,  # Numerical stability
                    max=self.max_logit,  # Numerical stability
                )
            ),
            min=self.min_clip,  # Numerical stability
            max=self.max_mu_clip,  # Numerical stability
        )  # Shape (N, G)
        nb_psi = torch.clamp(
            (
                torch.ones_like(library) @ self.b_psi  # Shape (N, 1) @ (1, G)
                + covariates_matrix @ self.w_psi  # Shape (N, B) @ (B, G)
            ),
            min=self.min_logit,  # Numerical stability
            max=self.max_logit,  # Numerical stability
        )  # Shape (N, G)
        nb_ksi = torch.clamp(
            nb_mu * torch.exp(-nb_psi),
            min=self.min_clip,  # Numerical stability
            max=self.max_ksi_clip,  # Numerical stability
        )  # Shape (N, G)
        zi_dropout_logits = self.decoder_dropouts(x_decoded)

        # Calculate NLL
        nll_loss = -ZeroInflatedNegativeBinomial(
            total_count=nb_ksi,  # Rate/overdispersion
            logits=nb_psi,  # Log-odds
            gate_logits=zi_dropout_logits,
            validate_args=False,
        ).log_prob(x).sum()

        # Return latent space
        return kld_loss, nll_loss

class PaddedEtude(Etude):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Save hyperparameters:
        assert padding_size >= 0, 'ERROR: padding_size must be non-negative'
        if padding_size == 0:
            print('WARNING: Do not use PaddedEtude if no padding used')
        self.padding_size = int(padding_size)  # Must be Python int
        self.padded_input_size = self.input_size + self.padding_size

        # Encoder layers
        layers = []
        layers.append(nn.Linear(self.padded_input_size, self.n_hidden))  # For now, just pad encoder input
        layers.append(nn.BatchNorm1d(self.n_hidden, eps=self.bn_eps, momentum=self.bn_moment))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(self.n_hidden, self.n_hidden))
            layers.append(nn.BatchNorm1d(self.n_hidden, eps=self.bn_eps, momentum=self.bn_moment))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_rate))
        self.encoder_layers = nn.Sequential(*layers)

        # Decoder layers
        self.decoder_recon = nn.Linear(self.n_hidden, self.padded_input_size)

        # Initialize GLM weights
        self.b_mu = nn.Parameter(torch.ones(1, self.padded_input_size))  # Shape (1, G)
        self.w_mu_gene = nn.Parameter(torch.ones(self.padded_input_size))  # Shape (G)
        self.w_mu_lib = nn.Parameter(torch.ones(1, self.padded_input_size))  # Shape (1, G)
        self.w_mu_cov = nn.Parameter(torch.zeros(self.cov_size, self.padded_input_size))  # Shape (B, G)
        self.b_psi = nn.Parameter(torch.ones(1, self.padded_input_size))  # Shape (1, G)
        self.w_psi = nn.Parameter(torch.zeros(self.cov_size, self.padded_input_size))  # Shape (B, G)

    def forward(self, x_aug):
        # Extract gene data and covariates from [X_genes; X_covariates]
        x = x_aug[:, :self.input_size]
        covariates_matrix = x_aug[:, self.input_size:]
        library = torch.sum(x, dim=1, keepdim=True)  # Shape (N, 1)

        # Log1p transformation for stability
        x_encoded = torch.log1p(x)  # Shape (N, G)

        # Pad for torch.compile memory address alignment
        x_encoded = F.pad(x_encoded, (0, self.padding_size), mode='constant', value=0)  # Shape (N, G + P)

        # Run inference
        x_encoded = self.encoder_layers(x_encoded)  # Shape (N, H)

        # Latent posterior distribution q(z | x)
        posterior_mu = self.encoder_mean(x_encoded)
        posterior_log_var = self.encoder_log_var(x_encoded)
        posterior_sigma = torch.clamp(
            torch.exp(0.5 * posterior_log_var), min=self.epsilon
        )  # Shape (N, Z)

        # Construct posterior distribution
        posterior_dist = Normal(posterior_mu, posterior_sigma)

        # Reparameterization trick
        posterior_latent = posterior_dist.rsample()  # Shape (N, Z)

        # Calculate KL divergence penalty
        prior_dist = Normal(torch.zeros_like(posterior_mu), torch.ones_like(posterior_mu))
        kld_loss = _kl_normal_normal(posterior_dist, prior_dist).sum()  # Shape (N, Z)

        # Run generative model
        x_decoded = torch.cat([posterior_latent, covariates_matrix], dim=1)  # Shape (N, Z + B)
        x_decoded = self.decoder_layers(x_decoded)  # Shape (N, H)
        x_bar = self.decoder_recon(x_decoded)  # Shape (N, G + P)
        x_bar = self.decoder_recon_act(x_bar[:, :self.input_size])  # Shape (N, G)
        x_bar = F.pad(x_bar, (0, self.padding_size), mode='constant', value=0)  # Shape (N, G + P)

        # Parameterize (ZI)NB
        nb_mu = torch.clamp(
            torch.exp(
                torch.clamp(
                    (
                        torch.ones_like(library) @ self.b_mu  # Shape (N, 1) @ (1, G)
                        + torch.log(torch.clamp(x_bar, min=self.min_clip)) * self.w_mu_gene  # Shape (N, G) * (G)
                        + torch.log(torch.clamp(library, min=self.min_library_size)) @ self.w_mu_lib  # Shape (N, 1) @ (1, G)
                        + covariates_matrix @ self.w_mu_cov  # Shape (N, B) @ (B, G)
                    ),
                    min=self.min_logit,  # Numerical stability
                    max=self.max_logit,  # Numerical stability
                )
            ),
            min=self.min_clip,  # Numerical stability
            max=self.max_mu_clip,  # Numerical stability
        )  # Shape (N, G)
        nb_psi = torch.clamp(
            (
                torch.ones_like(library) @ self.b_psi  # Shape (N, 1) @ (1, G)
                + covariates_matrix @ self.w_psi  # Shape (N, B) @ (B, G)
            ),
            min=self.min_logit,  # Numerical stability
            max=self.max_logit,  # Numerical stability
        )  # Shape (N, G)
        nb_ksi = torch.clamp(
            nb_mu * torch.exp(-nb_psi),
            min=self.min_clip,  # Numerical stability
            max=self.max_ksi_clip,  # Numerical stability
        )  # Shape (N, G)

        # Calculate NLL
        nll_loss = -NegativeBinomial(
            total_count=nb_ksi[:, :self.input_size],  # Rate/overdispersion
            logits=nb_psi[:, :self.input_size],  # Log-odds
            validate_args=False,
        ).log_prob(x).sum()

        # Return latent space
        return kld_loss, nll_loss

    def get_batch_latent_representation(self, x_aug, mc_samples=0):
        # Run inference
        x = x_aug[:, :self.input_size]
        x_encoded = torch.log1p(x)  # Shape (N, G)
        x_encoded = F.pad(x_encoded, (0, self.padding_size), mode='constant', value=0)  # Shape (N, G + P)
        x_encoded = self.encoder_layers(x_encoded)  # Shape (N, H)

        # Posterior distribution q(z | x)
        posterior_mu = self.encoder_mean(x_encoded)
        posterior_log_var = self.encoder_log_var(x_encoded)
        posterior_sigma = torch.clamp(
            torch.exp(0.5 * posterior_log_var), min=self.epsilon
        )  # Shape (N, Z)
        posterior_dist = Normal(posterior_mu, posterior_sigma)

        # Sample latent space representations
        if mc_samples > 0:
            posterior_latent_list = posterior_dist.sample([mc_samples])  # Shape (MC, N, Z)
            posterior_latent = torch.mean(posterior_latent_list, dim=0)  # Shape (N, Z)
        else:
            posterior_latent = posterior_dist.sample()  # Shape (N, Z)

        return posterior_latent

class PaddedZinbEtude(PaddedEtude):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder_dropouts = nn.Linear(self.n_hidden, self.padded_input_size)

    def forward(self, x_aug):
        # Extract gene data and covariates from [X_genes; X_covariates]
        x = x_aug[:, :self.input_size]
        covariates_matrix = x_aug[:, self.input_size:]
        library = torch.sum(x, dim=1, keepdim=True)  # Shape (N, 1)

        # Log1p transformation for stability
        x_encoded = torch.log1p(x)  # Shape (N, G)

        # Pad for torch.compile memory address alignment
        x_encoded = F.pad(x_encoded, (0, self.padding_size), mode='constant', value=0)  # Shape (N, G + P)

        # Run inference
        x_encoded = self.encoder_layers(x_encoded)  # Shape (N, H)

        # Latent posterior distribution q(z | x)
        posterior_mu = self.encoder_mean(x_encoded)
        posterior_log_var = self.encoder_log_var(x_encoded)
        posterior_sigma = torch.clamp(
            torch.exp(0.5 * posterior_log_var), min=self.epsilon
        )  # Shape (N, Z)

        # Construct posterior distribution
        posterior_dist = Normal(posterior_mu, posterior_sigma)

        # Reparameterization trick
        posterior_latent = posterior_dist.rsample()  # Shape (N, Z)

        # Calculate KL divergence penalty
        prior_dist = Normal(torch.zeros_like(posterior_mu), torch.ones_like(posterior_mu))
        kld_loss = _kl_normal_normal(posterior_dist, prior_dist).sum()  # Shape (N, Z)

        # Run generative model
        x_decoded = torch.cat([posterior_latent, covariates_matrix], dim=1)  # Shape (N, Z + B)
        x_decoded = self.decoder_layers(x_decoded)  # Shape (N, H)
        x_bar = self.decoder_recon(x_decoded)  # Shape (N, G + P)
        x_bar = self.decoder_recon_act(x_bar[:, :self.input_size])  # Shape (N, G)
        x_bar = F.pad(x_bar, (0, self.padding_size), mode='constant', value=0)  # Shape (N, G + P)

        # Parameterize (ZI)NB
        nb_mu = torch.clamp(
            torch.exp(
                torch.clamp(
                    (
                        torch.ones_like(library) @ self.b_mu  # Shape (N, 1) @ (1, G)
                        + torch.log(torch.clamp(x_bar, min=self.min_clip)) * self.w_mu_gene  # Shape (N, G) * (G)
                        + torch.log(torch.clamp(library, min=self.min_library_size)) @ self.w_mu_lib  # Shape (N, 1) @ (1, G)
                        + covariates_matrix @ self.w_mu_cov  # Shape (N, B) @ (B, G)
                    ),
                    min=self.min_logit,  # Numerical stability
                    max=self.max_logit,  # Numerical stability
                )
            ),
            min=self.min_clip,  # Numerical stability
            max=self.max_mu_clip,  # Numerical stability
        )  # Shape (N, G)
        nb_psi = torch.clamp(
            (
                torch.ones_like(library) @ self.b_psi  # Shape (N, 1) @ (1, G)
                + covariates_matrix @ self.w_psi  # Shape (N, B) @ (B, G)
            ),
            min=self.min_logit,  # Numerical stability
            max=self.max_logit,  # Numerical stability
        )  # Shape (N, G)
        nb_ksi = torch.clamp(
            nb_mu * torch.exp(-nb_psi),
            min=self.min_clip,  # Numerical stability
            max=self.max_ksi_clip,  # Numerical stability
        )  # Shape (N, G)
        zi_dropout_logits = self.decoder_dropouts(x_decoded)

        # Calculate NLL
        nll_loss = -ZeroInflatedNegativeBinomial(
            total_count=nb_ksi[:, :self.input_size],  # Rate/overdispersion
            logits=nb_psi[:, :self.input_size],  # Log-odds
            gate_logits=zi_dropout_logits[:, :self.input_size],
            validate_args=False,
        ).log_prob(x).sum()

        # Return latent space
        return kld_loss, nll_loss

class scVI(Etude):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # No build-level control flow determinism used
        self.decoder_dropouts = nn.Linear(self.n_hidden, self.input_size)

        # Only use one set of batch keys
        self.use_batch_keys = self.n_batch_keys > 1
        self.n_batch_keys = int(self.n_batch_keys) if self.n_batch_keys > 0 else 1  # Must be Python int
        self.w_psi = nn.Parameter(torch.zeros(self.n_batch_keys, self.input_size))  # Shape (B, G)

    def forward(self, x_aug):
        # Extract gene data and covariates from [X_genes; X_covariates]
        x = x_aug[:, :self.input_size]
        covariates_matrix = x_aug[:, self.input_size:]
        library = torch.sum(x, dim=1, keepdim=True)  # Shape (N, 1)
        if self.use_batch_keys:
            batch_keys = x_aug[:, self.input_size:self.input_size + self.n_batch_keys]

        # Run inference model
        x_encoded = torch.log1p(x)  # Shape (N, G)
        x_encoded = self.encoder_layers(x_encoded)  # Shape (N, H)

        # Latent posterior distribution q(z | x)
        posterior_mu = self.encoder_mean(x_encoded)
        posterior_log_var = self.encoder_log_var(x_encoded)
        posterior_sigma = torch.clamp(
            torch.exp(0.5 * posterior_log_var), min=self.epsilon
        )  # Shape (N, Z)

        # Construct posterior distribution
        posterior_dist = Normal(posterior_mu, posterior_sigma)
        posterior_latent = posterior_dist.rsample()  # Shape (N, Z)

        # Calculate KL divergence penalty
        prior_dist = Normal(torch.zeros_like(posterior_mu), torch.ones_like(posterior_mu))
        kld_loss = _kl_normal_normal(posterior_dist, prior_dist).sum()  # Shape (N, Z)

        # Run generative model
        x_decoded = torch.cat([posterior_latent, covariates_matrix], dim=1)  # Shape (N, Z + B)
        x_decoded = self.decoder_layers(x_decoded)  # Shape (N, H)
        x_bar = self.decoder_recon(x_decoded)  # Shape (N, G)
        x_bar = self.decoder_recon_act(x_bar)  # Shape (N, G)

        # Parameterize (ZI)NB
        nb_mu = torch.clamp(
            torch.exp(
                torch.clamp(
                    (
                        torch.log(torch.clamp(x_bar, min=self.min_clip))  # Shape (N, G)
                        + torch.log(torch.clamp(library, min=1))  # Shape (N, 1) will broadcast
                    ),
                    min=self.min_logit,  # Numerical stability
                    max=self.max_logit,  # Numerical stability
                )
            ),
            min=self.min_clip,  # Numerical stability
            max=self.max_mu_clip,  # Numerical stability
        )  # Shape (N, G)
        if self.use_batch_keys:
            nb_psi = torch.clamp(
                (
                    batch_keys @ self.w_psi  # Shape (N, B) @ (B, G)
                ),
                min=self.min_logit,  # Numerical stability
                max=self.max_logit,  # Numerical stability
            )  # Shape (N, G)
        else:
            nb_psi = torch.clamp(
                (
                    torch.ones_like(library) @ self.w_psi  # Shape (N, 1) @ (B=1, G)
                ),
                min=self.min_logit,  # Numerical stability
                max=self.max_logit,  # Numerical stability
            )  # Shape (N, G)
        nb_ksi = torch.clamp(
            nb_mu * torch.exp(-nb_psi),
            min=self.min_clip,  # Numerical stability
            max=self.max_ksi_clip,  # Numerical stability
        )  # Shape (N, G)
        zi_dropout_logits = self.decoder_dropouts(x_decoded) if self.dist == 'zinb' else None

        # Calculate NLL
        if self.dist == 'zinb':
            nll_loss = -ZeroInflatedNegativeBinomial(
                total_count=nb_ksi,  # Rate/overdispersion
                logits=nb_psi,  # Log-odds
                gate_logits=zi_dropout_logits,
            ).log_prob(x).sum()
        elif self.dist == 'nb':
            nll_loss = -NegativeBinomial(
                total_count=nb_ksi,  # Rate/overdispersion
                logits=nb_psi,  # Log-odds
            ).log_prob(x).sum()
        else:
            raise ValueError(f'Only ZINB and NB are supported, not: {self.dist}')

        # Return latent space
        return kld_loss, nll_loss
