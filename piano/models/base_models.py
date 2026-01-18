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

import torch
import torch.nn.functional as F
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from torch import nn
from torch.distributions import NegativeBinomial, Normal
from torch.distributions.kl import _kl_normal_normal

from piano.models.base_modules import grad_reverse


class Etude(nn.Module):
    def __init__(
        self,

        # Model architecture
        input_size: int = 4096,  # Must be Python int
        n_hidden: int = 256,
        n_layers: int = 3,
        latent_size: int = 32,
        n_total_covariate_dims: int = 0,
        n_categorical_covariate_dims: int = 0,

        # Model hyperparameters
        dropout_rate: float = 0.1,
        batchnorm_eps: float = 1e-5,       # Torch default is 1e-5
        batchnorm_momentum: float = 1e-1,  # Torch default is 1e-1
        epsilon: float = 1e-5,             # Torch default is 1e-5

        # Training mode
        padding_size: int = 0,  # Must be Python int
        adversarial: bool = True,
    ):
        super().__init__()

        # Save architecture
        self.input_size = int(input_size)  # Must be Python int
        self.n_hidden = int(n_hidden)
        self.n_layers = int(n_layers)
        self.latent_size = int(latent_size)
        self.n_total_covariate_dims = int(n_total_covariate_dims)
        self.n_categorical_covariate_dims = int(n_categorical_covariate_dims)
        self.padding_size = int(padding_size)

        # Save hyperparameters
        self.dropout_rate = dropout_rate
        self.bn_eps = batchnorm_eps
        self.bn_moment = batchnorm_momentum
        self.epsilon = epsilon

        # Training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Encoder layers
        layers = []
        layers.append(nn.Linear(self.input_size + self.padding_size, self.n_hidden))
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

        # Decoder layers
        layers = []
        layers.append(nn.Linear(self.latent_size + self.n_total_covariate_dims, self.n_hidden))
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
        self.w_mu_cov = nn.Parameter(torch.zeros(self.n_total_covariate_dims, self.input_size))  # Shape (B, G)
        self.b_psi = nn.Parameter(torch.ones(1, self.input_size))  # Shape (1, G)
        self.w_psi = nn.Parameter(torch.zeros(self.n_total_covariate_dims, self.input_size))  # Shape (B, G)

        # Numerical stability
        self.max_logit = 20
        self.min_logit = -20
        self.max_mu_clip = 2e4
        self.max_ksi_clip = 1e8
        self.min_clip = 1e-8
        self.min_library_size = 2e2

        # Initialize adversarial training mode
        self.adversarial = adversarial
        if self.adversarial:
            self.forward = self._forward_adv
            self.batch_classifier = nn.Sequential(
                nn.Linear(self.latent_size, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_categorical_covariate_dims),
            )
        else:
            self.forward = self._forward_no_adv
            self.batch_classifier = None

        # Toggle padding
        if self.padding_size > 0:
            self._prepare_encoder_input = self._prepare_encoder_input_with_padding
        else:
            self._prepare_encoder_input = self._prepare_encoder_input_without_padding

    def _parse_augmented_matrix(self, x_aug):
        # Extract gene data and covariates from [X_genes; X_covariates]
        x_raw = x_aug[:, :self.input_size]
        covariates_matrix = x_aug[:, self.input_size:]
        categorical_covariates_matrix = x_aug[:, self.input_size:self.input_size + self.n_categorical_covariate_dims]

        # Save library size for to scale decoder softmax output for reconstruction
        library = torch.sum(x_raw, dim=1, keepdim=True)  # Shape (N, 1)

        return x_raw, covariates_matrix, categorical_covariates_matrix, library

    def _prepare_encoder_input_without_padding(self, x_raw):
        # Log1p transformation for stability
        return torch.log1p(x_raw)  # Shape (N, G)

    def _prepare_encoder_input_with_padding(self, x_raw):
        # Log1p transformation for stability
        return F.pad(torch.log1p(x_raw), (0, self.padding_size), value=0)  # Shape (N, G + P)

    def _encode_latent(self, x):
        # Run inference
        x_encoded = self.encoder_layers(x)  # Shape (N, H)

        # Latent posterior distribution q(z | x)
        posterior_mu = self.encoder_mean(x_encoded)
        posterior_log_var = self.encoder_log_var(x_encoded)
        posterior_sigma = torch.clamp(
            torch.exp(0.5 * posterior_log_var), min=self.epsilon
        )  # Shape (N, Z)

        # Construct posterior distribution
        posterior_dist = Normal(posterior_mu, posterior_sigma)

        return posterior_dist

    def _decode_latent(self, posterior_latent, covariates_matrix):
        x_decoded = torch.cat([posterior_latent, covariates_matrix], dim=1)  # Shape (N, Z + B)
        x_decoded = self.decoder_layers(x_decoded)  # Shape (N, H)
        x_bar = self.decoder_recon(x_decoded)  # Shape (N, G)
        x_bar = self.decoder_recon_act(x_bar)  # Shape (N, G)

        return x_bar

    def _nb_mu(self, x_bar, library, covariates_matrix):
        return torch.clamp(
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

    def _nb_psi(self, library, covariates_matrix):
        return torch.clamp(
            (
                torch.ones_like(library) @ self.b_psi  # Shape (N, 1) @ (1, G)
                + covariates_matrix @ self.w_psi  # Shape (N, B) @ (B, G)
            ),
            min=self.min_logit,  # Numerical stability
            max=self.max_logit,  # Numerical stability
        )  # Shape (N, G)

    def _nb_ksi(self, nb_mu, nb_psi):
        return torch.clamp(
            nb_mu * torch.exp(-nb_psi),
            min=self.min_clip,  # Numerical stability
            max=self.max_ksi_clip,  # Numerical stability
        )  # Shape (N, G)

    def _kld_loss(self, posterior_latent, posterior_dist):
        prior_dist = Normal(torch.zeros_like(posterior_latent), torch.ones_like(posterior_latent))
        return _kl_normal_normal(posterior_dist, prior_dist).sum()  # Shape (N, Z)

    def _nll_loss(self, nb_ksi, nb_psi, x):
        return -NegativeBinomial(
            total_count=nb_ksi,  # Rate/overdispersion
            logits=nb_psi,  # Log-odds
            validate_args=False,
        ).log_prob(x).sum()

    def _adv_loss(self, posterior_latent, categorical_covariates_matrix):
        z_adv = grad_reverse(posterior_latent)
        batch_logits = self.batch_classifier(z_adv)
        return F.binary_cross_entropy_with_logits(batch_logits, categorical_covariates_matrix, reduction='sum')

    def _forward_adv(self, x_aug):
        # Parse augmented matrix
        x_raw, covariates_matrix, categorical_covariates_matrix, library = self._parse_augmented_matrix(x_aug)
        x_log_padded = self._prepare_encoder_input(x_raw)
        # Encode data to isotropic Gaussian latent space
        posterior_dist = self._encode_latent(x_log_padded)  # Normal(posterior_mu, posterior_sigma)
        # Reparameterization trick
        posterior_latent = posterior_dist.rsample()  # Shape (N, Z)
        # Run generative model
        x_bar = self._decode_latent(posterior_latent, covariates_matrix)
        # Parameterize (ZI)NB
        nb_mu = self._nb_mu(x_bar, library, covariates_matrix)
        nb_psi = self._nb_psi(library, covariates_matrix)
        nb_ksi = self._nb_ksi(nb_mu, nb_psi)
        # Calculate losses
        kld_loss = self._kld_loss(posterior_latent, posterior_dist)
        nll_loss = self._nll_loss(nb_ksi, nb_psi, x_raw)
        adv_loss = self._adv_loss(posterior_latent, categorical_covariates_matrix)

        return {'nll': nll_loss, 'kld': kld_loss, 'adv': adv_loss}

    def _forward_no_adv(self, x_aug):
        # Parse augmented matrix
        x_raw, covariates_matrix, categorical_covariates_matrix, library = self._parse_augmented_matrix(x_aug)
        x_log_padded = self._prepare_encoder_input(x_raw)
        # Encode data to isotropic Gaussian latent space
        posterior_dist = self._encode_latent(x_log_padded)  # Normal(posterior_mu, posterior_sigma)
        # Reparameterization trick
        posterior_latent = posterior_dist.rsample()  # Shape (N, Z)
        # Run generative model
        x_bar = self._decode_latent(posterior_latent, covariates_matrix)
        # Parameterize (ZI)NB
        nb_mu = self._nb_mu(x_bar, library, covariates_matrix)
        nb_psi = self._nb_psi(library, covariates_matrix)
        nb_ksi = self._nb_ksi(nb_mu, nb_psi)
        # Calculate losses
        kld_loss = self._kld_loss(posterior_latent, posterior_dist)
        nll_loss = self._nll_loss(nb_ksi, nb_psi, x_raw)

        return {'nll': nll_loss, 'kld': kld_loss, 'adv': 0}

    def training_step(self, batch, kld_weight, adv_weight):
        losses_dict = self.forward(batch)
        nll_loss, kld_loss, adv_loss = losses_dict['nll'], losses_dict['kld'], losses_dict['adv']
        elbo_loss = (nll_loss + kld_loss * kld_weight) / (1 + kld_weight)
        total_loss = elbo_loss + adv_loss * adv_weight

        return {'total': total_loss, 'elbo': elbo_loss, 'nll': nll_loss, 'kld': kld_loss, 'adv': adv_loss}

    def get_batch_latent_representation(self, x_aug, mc_samples=0):
        # Parse augmented matrix
        x_raw, covariates_matrix, categorical_covariates_matrix, library = self._parse_augmented_matrix(x_aug)
        x_log_padded = self._prepare_encoder_input(x_raw)
        # Encode data to isotropic Gaussian latent space
        posterior_dist = self._encode_latent(x_log_padded)  # Normal(posterior_mu, posterior_sigma)
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
        # Parse augmented matrix
        x_raw, original_covariates_matrix, original_categorical_covariates_matrix, library = self._parse_augmented_matrix(x_aug)
        x_log_padded = self._prepare_encoder_input(x_raw)
        if covariates_matrix is None:
            covariates_matrix = original_covariates_matrix
        # Encode data to isotropic Gaussian latent space
        posterior_dist = self._encode_latent(x_log_padded)  # Normal(posterior_mu, posterior_sigma)
        # Sample latent space representations
        posterior_latent = posterior_dist.sample()  # Shape (N, Z)
        # Run generative model
        x_bar = self._decode_latent(posterior_latent, covariates_matrix)
        # Parameterize (ZI)NB
        nb_mu = self._nb_mu(x_bar, library, covariates_matrix)  # Shape (N, G)

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
        self.w_mu_cov = nn.Parameter(torch.zeros(self.n_total_covariate_dims, self.padded_input_size))  # Shape (B, G)
        self.b_psi = nn.Parameter(torch.ones(1, self.padded_input_size))  # Shape (1, G)
        self.w_psi = nn.Parameter(torch.zeros(self.n_total_covariate_dims, self.padded_input_size))  # Shape (B, G)

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
