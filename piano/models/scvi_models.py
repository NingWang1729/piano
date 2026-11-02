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
from torch import nn
from torch.distributions import Normal

from piano.models.base_models import Etude


class EtudeSCVI(nn.Module):
    """
    Variational autoencoder for raw counts: the same model as scvi-tools 1.4.2's VAE with an NB likelihood, one inverse
    dispersion per gene and batch, categorical covariates, the observed library size and a normal latent. With the same
    weights it gives scvi-tools' results to float tolerance.
    """

    # scvi-tools key prefixes and this model's names for them
    SCVI_RENAMES = {
        'px_r': 'log_dispersion',
        'z_encoder.encoder.fc_layers.Layer ': 'encoder_layers.',
        'z_encoder.mean_encoder.': 'encoder_mean.',
        'z_encoder.var_encoder.': 'encoder_log_var.',
        'decoder.px_decoder.fc_layers.Layer ': 'decoder_layers.',
        'decoder.px_scale_decoder.0.': 'decoder_recon.',
    }
    # scvi-tools key prefixes of the parts unused with these settings: library encoder, dispersion and dropout heads
    SCVI_UNUSED = ('l_encoder.', 'decoder.px_r_decoder.', 'decoder.px_dropout_decoder.')

    def __init__(
        self,
        input_size: int,
        n_batch: int,
        n_cats_per_cov=(),
        n_hidden: int = 128,
        latent_size: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Save architecture (Must be Python ints)
        self.input_size = int(input_size)
        self.n_batch = int(n_batch)
        self.n_cats_per_cov = tuple(int(n) for n in n_cats_per_cov)
        self.n_hidden = int(n_hidden)
        self.latent_size = int(latent_size)
        self.n_layers = int(n_layers)
        self.dropout_rate = dropout_rate

        # Decoder covariates: one-hot batch and categorical covariates, leaving out single-category ones
        self.n_covariate_dims = sum(n for n in (self.n_batch, *self.n_cats_per_cov) if n > 1)

        # Columns of the decoder covariates in PIANO's augmented matrix, which has a column for every category; not saved
        # in the state_dict, as they follow from the architecture
        self.register_buffer('decoder_covariate_columns', self._decoder_covariate_columns(), persistent=False)

        # Training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Batch normalisation settings of scvi-tools
        self.bn_eps = 1e-3
        self.bn_moment = 1e-2

        # Numerical stability: scvi-tools' var_eps (posterior variance) and the eps of its NB log likelihood
        self.var_eps = 1e-4
        self.nb_eps = 1e-8

        # Log inverse dispersion per gene and batch
        self.log_dispersion = nn.Parameter(torch.randn(self.input_size, self.n_batch))  # Shape (G, B)

        # Encoder layers: one block per hidden layer
        blocks = []
        for i in range(self.n_layers):
            in_size = self.input_size if i == 0 else self.n_hidden
            blocks.append(self._hidden_block(in_size, dropout_rate=self.dropout_rate))
        self.encoder_layers = nn.ModuleList(blocks)
        self.encoder_mean = nn.Linear(self.n_hidden, self.latent_size)
        self.encoder_log_var = nn.Linear(self.n_hidden, self.latent_size)

        # Decoder layers: one block per hidden layer, each fed the covariates too
        blocks = []
        for i in range(self.n_layers):
            in_size = (self.latent_size if i == 0 else self.n_hidden) + self.n_covariate_dims
            blocks.append(self._hidden_block(in_size, dropout_rate=0.0))
        self.decoder_layers = nn.ModuleList(blocks)
        self.decoder_recon = nn.Linear(self.n_hidden, self.input_size)
        self.decoder_recon_act = nn.Softmax(dim=-1)

    def _decoder_covariate_columns(self):
        """
        Indices, in PIANO's augmented matrix [counts | one-hot batch | one-hot of each other categorical covariate], of
        the decoder covariates: every categorical block except those of a single category. Shape (C,).
        """
        columns, start = [], self.input_size
        for n_categories in (self.n_batch, *self.n_cats_per_cov):
            if n_categories > 1:
                columns.append(torch.arange(start, start + n_categories))
            start += n_categories
        return torch.cat(columns) if columns else torch.zeros(0, dtype=torch.long)

    def _hidden_block(self, in_size, dropout_rate):
        """Linear, batch norm and ReLU, then dropout only if its rate is positive (as scvi-tools)."""
        layers = [
            nn.Linear(in_size, self.n_hidden),
            nn.BatchNorm1d(self.n_hidden, eps=self.bn_eps, momentum=self.bn_moment),
            nn.ReLU(),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        return nn.Sequential(*layers)

    def posterior(self, x):
        """Latent posterior q(z | x) from counts x of shape (N, G); its mean is the latent representation."""
        # Log1p transformation for stability
        x_encoded = torch.log1p(x)  # Shape (N, G)

        # Run inference
        for block in self.encoder_layers:
            x_encoded = block(x_encoded)  # Shape (N, H)

        # Variance = exp(raw output) + var_eps, as scvi-tools, so it stays positive
        posterior_mu = self.encoder_mean(x_encoded)  # Shape (N, Z)
        posterior_var = torch.exp(self.encoder_log_var(x_encoded)) + self.var_eps  # Shape (N, Z)
        return Normal(posterior_mu, posterior_var.sqrt())

    def _decode_latent(self, posterior_latent, covariates):
        """Expression proportions per gene: the covariates are appended to the input of every hidden block."""
        x_decoded = posterior_latent  # Shape (N, Z)
        for block in self.decoder_layers:
            x_decoded = block(torch.cat([x_decoded, covariates], dim=-1))  # Shape (N, H)
        x_bar = self.decoder_recon(x_decoded)  # Shape (N, G)
        return self.decoder_recon_act(x_bar)  # Shape (N, G)

    def _nb_log_likelihood(self, x, mu, theta):
        """
        NB log likelihood per cell and gene with mean mu and inverse dispersion theta, the log of
        Gamma(x + theta) / (Gamma(theta) x!) * (theta / (theta + mu))^theta * (mu / (theta + mu))^x:
            theta * log(theta / (theta + mu)) + x * log(mu / (theta + mu))
            + lgamma(x + theta) - lgamma(theta) - lgamma(x + 1),
        with each log guarded by a small constant.
        """
        log_theta_mu = torch.log(theta + mu + self.nb_eps)  # Shape (N, G)
        return (
            theta * (torch.log(theta + self.nb_eps) - log_theta_mu)
            + x * (torch.log(mu + self.nb_eps) - log_theta_mu)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )  # Shape (N, G)

    @staticmethod
    def _kl_to_standard_normal(posterior_dist):
        """
        KL(q(z|x) || N(0, I)) of a normal posterior q = N(mu, sigma^2) per cell, summed over latent dimensions. Per
        dimension, E_q[log q - log p] = (-log sigma - 1/2) - (-(sigma^2 + mu^2) / 2), so
            0.5 * (sigma^2 + mu^2 - 1) - log(sigma).
        The same as torch.distributions.kl_divergence, in closed form: its registry lookup breaks torch.compile's graph.
        """
        mu, sigma = posterior_dist.loc, posterior_dist.scale  # Shape (N, Z)
        return (0.5 * (sigma.square() + mu.square() - 1) - torch.log(sigma)).sum(dim=-1)  # Shape (N,)

    def loss_terms(self, x, batch_one_hot, covariates, kl_weight):
        """
        Loss of one minibatch: counts x (N, G), one-hot batch (N, B), decoder covariates (N, C) and the KL weight.
        Returns the mean loss and, per cell, the reconstruction loss and the KL divergence (as scvi-tools' metrics).
        """
        # Observed library size
        library = x.sum(dim=1, keepdim=True)  # Shape (N, 1)

        # Encode, then reparameterization trick
        posterior_dist = self.posterior(x)
        posterior_latent = posterior_dist.rsample()  # Shape (N, Z)

        # Run generative model
        x_bar = self._decode_latent(posterior_latent, covariates)  # Shape (N, G)

        # Parameterize NB: mean from the library size, inverse dispersion from the batch
        nb_mu = library * x_bar  # Shape (N, G)
        nb_theta = torch.exp(batch_one_hot @ self.log_dispersion.T)  # Shape (N, B) @ (B, G)

        # Per-cell losses
        reconstruction = -self._nb_log_likelihood(x, nb_mu, nb_theta).sum(dim=-1)  # Shape (N,)
        kl = self._kl_to_standard_normal(posterior_dist)  # Shape (N,)

        loss = torch.mean(reconstruction + kl_weight * kl)
        return {'loss': loss, 'reconstruction': reconstruction, 'kl': kl}

    def _parse_augmented_matrix(self, x_aug):
        """
        Counts (N, G), one-hot batch (N, B) and decoder covariates (N, C) from PIANO's augmented matrix. Columns after
        the categorical blocks (continuous covariates) are not read: scVI mode rejects them at Composer creation.
        """
        x = x_aug[:, :self.input_size]
        batch_one_hot = x_aug[:, self.input_size:self.input_size + self.n_batch]
        covariates = x_aug[:, self.decoder_covariate_columns]
        return x, batch_one_hot, covariates

    def training_step(self, x_aug, kld_weight, adv_weight=None):
        """
        Losses of one minibatch of PIANO's augmented matrix: 'loss', the mean loss to minimise, and the per-batch sums of
        the per-cell 'reconstruction' and 'kl' terms, for epoch averages. adv_weight is accepted so the call matches
        Etude.training_step's signature, and unused: this model has no adversary.
        """
        terms = self.loss_terms(*self._parse_augmented_matrix(x_aug), kld_weight)
        return {'loss': terms['loss'], 'reconstruction': terms['reconstruction'].sum(), 'kl': terms['kl'].sum()}

    def get_batch_latent_representation(self, x_aug, mc_samples=0):
        """Posterior mean (mc_samples=0) or the average of mc_samples posterior draws of each cell, as a numpy array."""
        posterior_dist = self.posterior(x_aug[:, :self.input_size])
        posterior_latent = torch.mean(Etude._posterior_latents(posterior_dist, mc_samples), dim=0)  # Shape (N, Z)
        return posterior_latent.cpu().numpy()

    def get_latent_representation(self, dataloader, mc_samples=0):
        """Latent representation of every cell of the dataloader's augmented batches, in eval mode, as Etude's."""
        return Etude.get_latent_representation(self, dataloader, mc_samples=mc_samples)

    @staticmethod
    def state_dict_from_scvi(state_dict):
        """
        This model's state_dict from that of a scvi-tools VAE built with these settings: used tensors are renamed,
        those of the unused parts are dropped. Raises ValueError on a key of neither kind (another scvi-tools setup).
        """
        converted = {}
        for key, tensor in state_dict.items():
            if key.startswith(EtudeSCVI.SCVI_UNUSED):
                continue
            prefix = next((p for p in EtudeSCVI.SCVI_RENAMES if key.startswith(p)), None)
            if prefix is None:
                raise ValueError(
                    f'Unexpected scvi-tools key {key!r}: build the VAE with gene_likelihood="nb", '
                    f'dispersion="gene-batch" and the observed library size'
                )
            converted[EtudeSCVI.SCVI_RENAMES[prefix] + key[len(prefix):]] = tensor
        return converted

    def load_scvi_state_dict(self, state_dict):
        """Load the weights of a scvi-tools VAE built with these settings."""
        self.load_state_dict(self.state_dict_from_scvi(state_dict), strict=True)
