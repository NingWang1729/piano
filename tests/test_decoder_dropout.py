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
from piano.models.base_models import Etude, EtudeMuTheta

N_CELLS = 300
N_GENES = 50
SMALL_BATCH = 32  # Tests on tiny data pass a small batch_size
# A small model: these tests check which dropout probabilities reach which layers, not model quality
SMALL_MODEL = {'n_hidden': 16, 'n_layers': 2, 'latent_size': 4}


# ---------------------------------------------------------------------------
# Helpers: small count matrix with a batch covariate, and a model's dropout probabilities
# ---------------------------------------------------------------------------
def make_adata(n_cells=N_CELLS, seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({'batch': pd.Categorical(rng.choice(['a', 'b'], n_cells))}, index=[str(i) for i in range(n_cells)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(n_cells, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(tmp_path, **kwargs):
    return Composer(make_adata(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'],
                    n_top_genes=-1, batch_size=SMALL_BATCH, early_stopping=False, deterministic=False,
                    outdir=str(tmp_path), **SMALL_MODEL, **kwargs)


def prepared_model(tmp_path, **kwargs):
    """The model of a Composer with prepared data."""
    pianist = make_composer(tmp_path, **kwargs)
    pianist.prepare_data()
    return pianist.prepare_model()


def dropout_probabilities(layers):
    """The p of every Dropout layer in an nn.Sequential."""
    return [layer.p for layer in layers if isinstance(layer, torch.nn.Dropout)]


# ---------------------------------------------------------------------------
# Defaults and explicit values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('dropout_rate', [0.1, 0.2])
def test_default_decoder_uses_dropout_rate(tmp_path, dropout_rate):
    # Coverage (passes on master): without decoder_dropout_rate, the decoder uses dropout_rate like the encoder, as before
    model = prepared_model(tmp_path, dropout_rate=dropout_rate)
    assert dropout_probabilities(model.decoder_layers) == [dropout_rate] * model.n_layers
    assert dropout_probabilities(model.encoder_layers) == [dropout_rate] * model.n_layers


@pytest.mark.parametrize('parameterization', ['ksi-psi', 'mu-theta'])
def test_decoder_dropout_rate_is_used_by_the_decoder_only(tmp_path, parameterization):
    model = prepared_model(tmp_path, parameterization=parameterization, dropout_rate=0.2, decoder_dropout_rate=0.05)
    assert dropout_probabilities(model.decoder_layers) == [0.05] * model.n_layers
    assert dropout_probabilities(model.encoder_layers) == [0.2] * model.n_layers


@pytest.mark.parametrize('model_class', [Etude, EtudeMuTheta])
def test_model_classes_default_to_dropout_rate_in_the_decoder(model_class):
    # Coverage (passes on master): users who build the model directly get the Composer's default
    model = model_class(input_size=N_GENES, **SMALL_MODEL, dropout_rate=0.2)
    assert dropout_probabilities(model.decoder_layers) == [0.2] * model.n_layers


@pytest.mark.parametrize('model_class', [Etude, EtudeMuTheta])
def test_model_classes_use_decoder_dropout_rate_in_the_decoder_only(model_class):
    model = model_class(input_size=N_GENES, **SMALL_MODEL, decoder_dropout_rate=0.0)
    assert dropout_probabilities(model.decoder_layers) == [0.0] * model.n_layers
    assert dropout_probabilities(model.encoder_layers) == [0.1] * model.n_layers


# ---------------------------------------------------------------------------
# Invalid values raise at creation, before any data are prepared
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('value', [-0.1, 1.5, float('nan'), '0.1', True])
def test_invalid_decoder_dropout_rate_raises_at_creation(tmp_path, value):
    with pytest.raises(ValueError, match=r'decoder_dropout_rate must be None or a number in \[0, 1\]'):
        make_composer(tmp_path, decoder_dropout_rate=value)


@pytest.mark.parametrize('value, expected', [(None, 0.2), (0, 0), (1, 1), (np.float32(0.25), 0.25)])
def test_valid_decoder_dropout_rate_edges_are_accepted(tmp_path, value, expected):
    # None passed explicitly means the decoder uses dropout_rate
    model = prepared_model(tmp_path, dropout_rate=0.2, decoder_dropout_rate=value)
    assert dropout_probabilities(model.decoder_layers) == [expected] * model.n_layers


# ---------------------------------------------------------------------------
# Training end to end
# ---------------------------------------------------------------------------
def test_one_epoch_trains_with_decoder_dropout_rate(tmp_path):
    # Cheap end-to-end check that the option trains and infers
    pianist = make_composer(tmp_path, decoder_dropout_rate=0.0, max_epochs=1)
    pianist.run_pipeline()
    assert pianist.get_latent_representation(make_adata()).shape == (N_CELLS, SMALL_MODEL['latent_size'])
