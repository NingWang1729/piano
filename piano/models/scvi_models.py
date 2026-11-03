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
from torch import nn
from torch.distributions import Normal

from piano.models.base_models import Etude


class EtudeSCVI(nn.Module):
    """
    Variational autoencoder for raw counts: the same model as scvi-tools 1.4.2's VAE with an NB (default) or ZINB
    likelihood, one inverse dispersion per gene and batch, categorical and continuous covariates, the observed library
    size and a normal latent. With the same weights it gives scvi-tools' results to float tolerance.

    Covariates enter the decoder only, as scvi-tools' defaults: the one-hot batch and categorical covariates are appended
    to the input of every decoder layer, the continuous covariates (raw values, not rescaled) to the latent only, at the
    decoder's input.
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
    # scvi-tools key prefix of the dropout head (ZINB only) and this model's name for it, as Etude's ZINB head
    SCVI_ZINB_RENAMES = {'decoder.px_dropout_decoder.': 'decoder_dropouts.'}
    # scvi-tools key prefixes of the parts unused with these settings: library encoder, dispersion and (with NB) dropout heads
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
        distribution: str = 'nb',
        n_continuous_cov: int = 0,
    ):
        super().__init__()

        # Likelihood of the counts: NB, or ZINB with a dropout head
        EtudeSCVI._check_distribution(distribution)
        self.distribution = distribution

        # Save architecture (Must be Python ints)
        self.input_size = int(input_size)
        self.n_batch = int(n_batch)
        self.n_cats_per_cov = tuple(int(n) for n in n_cats_per_cov)
        self.n_hidden = int(n_hidden)
        self.latent_size = int(latent_size)
        self.n_layers = int(n_layers)
        self.dropout_rate = dropout_rate
        self.n_continuous_cov = int(n_continuous_cov)

        # Decoder covariates: one-hot batch and categorical covariates, leaving out single-category ones
        self.n_covariate_dims = sum(n for n in (self.n_batch, *self.n_cats_per_cov) if n > 1)

        # Columns of the decoder covariates in PIANO's augmented matrix, which has a column for every category; not saved
        # in the state_dict, as they follow from the architecture
        self.register_buffer('decoder_covariate_columns', self._decoder_covariate_columns(), persistent=False)

        # First column of the continuous covariates in PIANO's augmented matrix: after the counts and every categorical
        # block (single-category ones included)
        self.continuous_start = self.input_size + self.n_batch + sum(self.n_cats_per_cov)

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

        # Decoder layers: one block per hidden layer, each fed the categorical covariates too; the first also reads the
        # continuous covariates, appended to the latent
        blocks = []
        for i in range(self.n_layers):
            in_size = (self.latent_size + self.n_continuous_cov if i == 0 else self.n_hidden) + self.n_covariate_dims
            blocks.append(self._hidden_block(in_size, dropout_rate=0.0))
        self.decoder_layers = nn.ModuleList(blocks)
        self.decoder_recon = nn.Linear(self.n_hidden, self.input_size)
        self.decoder_recon_act = nn.Softmax(dim=-1)

        # ZINB only: zero-inflation (dropout) logits per gene from the decoder's last hidden layer (no activation); not
        # dropout layers, which this decoder does not have
        if self.distribution == 'zinb':
            self.decoder_dropouts = nn.Linear(self.n_hidden, self.input_size)

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

    def _decode_hidden(self, posterior_latent, covariates, continuous_covariates):
        """
        Decoder's last hidden activations: the continuous covariates (N, K) are appended to the latent, as scvi-tools'
        decoder input, then the categorical covariates (N, C) to the input of every hidden block.
        """
        x_decoded = torch.cat([posterior_latent, continuous_covariates], dim=-1)  # Shape (N, Z + K)
        for block in self.decoder_layers:
            x_decoded = block(torch.cat([x_decoded, covariates], dim=-1))  # Shape (N, H)
        return x_decoded

    def _expression_proportions(self, x_decoded):
        """Expression proportions per gene from the decoder's last hidden activations."""
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

    def _zinb_log_likelihood(self, x, mu, theta, zi_dropout_logits):
        """
        ZINB log likelihood per cell and gene.

        A count comes from one of two sources: with probability pi it is an extra zero beyond what the NB predicts
        (often read as technical dropout), otherwise it is drawn from NB(mu, theta), which can also give a zero by
        chance. So a positive count must come from the NB, and a zero can come from either:
            x > 0: P(x) = (1 - pi) * NB(x)
            x = 0: P(0) = pi + (1 - pi) * NB(0),   NB(0) = (theta / (theta + mu))^theta

        The model gives pi as a logit l, zi_dropout_logits (pi = sigmoid(l) = e^l / (1 + e^l)). Computing e^l can
        overflow, and NB(0) can be too small to represent, so both cases are rewritten in log space with
        softplus(z) = log(1 + e^z), which torch computes without overflow:
            log(1 - pi) = -softplus(l), so
            x > 0: log NB(x) - softplus(l)
            x = 0: log(e^l + NB(0)) - softplus(l)                (common denominator 1 + e^l)
                 = l + softplus(log NB(0) - l) - softplus(l)      (e^l taken out of the log; only log NB(0) is needed)
                 = softplus(log NB(0) - l) - softplus(-l)         (as softplus(l) - l = softplus(-l))

        Both cases are computed for every entry and one is picked per entry with a mask (x < nb_eps counts as zero),
        rather than branching on the data, so the compiled graph stays fixed.
        """
        # Log probability of a zero count under the NB, with each log guarded by a small constant
        nb_log_zero = theta * (torch.log(theta + self.nb_eps) - torch.log(theta + mu + self.nb_eps))  # Shape (N, G)

        # The two cases, then the one of each entry
        case_zero = (F.softplus(nb_log_zero - zi_dropout_logits)
                     - F.softplus(-zi_dropout_logits))  # Shape (N, G)
        case_positive = self._nb_log_likelihood(x, mu, theta) - F.softplus(zi_dropout_logits)  # Shape (N, G)
        # x < nb_eps is the zero-count test, as in scvi-tools
        return torch.where(x < self.nb_eps, case_zero, case_positive)  # Shape (N, G)

    def _log_likelihood(self, x, mu, theta, x_decoded):
        """Log likelihood per cell and gene of the model's distribution; ZINB reads its dropout logits from x_decoded."""
        if self.distribution == 'zinb':
            return self._zinb_log_likelihood(x, mu, theta, self.decoder_dropouts(x_decoded))
        return self._nb_log_likelihood(x, mu, theta)

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

    def loss_terms(self, x, batch_one_hot, covariates, kl_weight, continuous_covariates=None):
        """
        Loss of one minibatch: counts x (N, G), one-hot batch (N, B), decoder covariates (N, C), the KL weight and the
        raw continuous covariates (N, K; None when the model has none). Returns the mean loss and, per cell, the
        reconstruction loss and the KL divergence (as scvi-tools' metrics).
        """
        if continuous_covariates is None:
            continuous_covariates = x.new_zeros(x.shape[0], 0)  # Shape (N, 0)
        if continuous_covariates.shape[1] != self.n_continuous_cov:
            raise ValueError(f'Expected {self.n_continuous_cov} continuous covariates, got {continuous_covariates.shape[1]}')

        # Observed library size
        library = x.sum(dim=1, keepdim=True)  # Shape (N, 1)

        # Encode, then reparameterization trick
        posterior_dist = self.posterior(x)
        posterior_latent = posterior_dist.rsample()  # Shape (N, Z)

        # Run generative model
        x_decoded = self._decode_hidden(posterior_latent, covariates, continuous_covariates)  # Shape (N, H)
        x_bar = self._expression_proportions(x_decoded)  # Shape (N, G)

        # Parameterize NB: mean from the library size, inverse dispersion from the batch
        nb_mu = library * x_bar  # Shape (N, G)
        nb_theta = torch.exp(batch_one_hot @ self.log_dispersion.T)  # Shape (N, B) @ (B, G)

        # Per-cell losses
        reconstruction = -self._log_likelihood(x, nb_mu, nb_theta, x_decoded).sum(dim=-1)  # Shape (N,)
        kl = self._kl_to_standard_normal(posterior_dist)  # Shape (N,)

        loss = torch.mean(reconstruction + kl_weight * kl)
        return {'loss': loss, 'reconstruction': reconstruction, 'kl': kl}

    def _parse_augmented_matrix(self, x_aug):
        """
        Counts (N, G), one-hot batch (N, B), decoder covariates (N, C) and continuous covariates (N, K) from PIANO's
        augmented matrix [counts | one-hot batch | one-hot of each other categorical covariate | continuous covariates].
        """
        x = x_aug[:, :self.input_size]
        batch_one_hot = x_aug[:, self.input_size:self.input_size + self.n_batch]
        covariates = x_aug[:, self.decoder_covariate_columns]
        continuous_covariates = x_aug[:, self.continuous_start:self.continuous_start + self.n_continuous_cov]
        return x, batch_one_hot, covariates, continuous_covariates

    def training_step(self, x_aug, kld_weight, adv_weight=None):
        """
        Losses of one minibatch of PIANO's augmented matrix: 'loss', the mean loss to minimise, and the per-batch sums of
        the per-cell 'reconstruction' and 'kl' terms, for epoch averages. adv_weight is accepted so the call matches
        Etude.training_step's signature, and unused: this model has no adversary.
        """
        x, batch_one_hot, covariates, continuous_covariates = self._parse_augmented_matrix(x_aug)
        terms = self.loss_terms(x, batch_one_hot, covariates, kld_weight, continuous_covariates)
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
    def _check_distribution(distribution):
        """Raise NotImplementedError unless distribution is 'nb' or 'zinb', the likelihoods this model has."""
        if distribution not in ('nb', 'zinb'):
            raise NotImplementedError(f"distribution must be 'nb' or 'zinb', not {distribution!r}")

    @staticmethod
    def state_dict_from_scvi(state_dict, distribution='nb'):
        """
        This model's state_dict, for the given distribution, from that of a scvi-tools VAE built with these settings:
        used tensors are renamed, those of the unused parts are dropped. Raises ValueError on a key of neither kind
        (another scvi-tools setup).
        """
        # With ZINB the dropout head is used: renamed before the unused parts are dropped
        EtudeSCVI._check_distribution(distribution)
        renames = {**EtudeSCVI.SCVI_RENAMES, **(EtudeSCVI.SCVI_ZINB_RENAMES if distribution == 'zinb' else {})}

        converted = {}
        for key, tensor in state_dict.items():
            prefix = next((p for p in renames if key.startswith(p)), None)
            if prefix is not None:
                converted[renames[prefix] + key[len(prefix):]] = tensor
            elif not key.startswith(EtudeSCVI.SCVI_UNUSED):
                raise ValueError(
                    f'Unexpected scvi-tools key {key!r}: build the VAE with gene_likelihood={distribution!r}, '
                    f'dispersion="gene-batch" and the observed library size'
                )
        return converted

    def load_scvi_state_dict(self, state_dict):
        """Load the weights of a scvi-tools VAE built with these settings and this model's distribution."""
        self.load_state_dict(self.state_dict_from_scvi(state_dict, self.distribution), strict=True)
