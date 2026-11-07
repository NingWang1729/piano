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

# Query cells whose categorical covariate (or scVI mode's batch) has a value not seen in training (SCRUM-19): latents
# warn and encode them as unlabeled, since the encoder does not use covariates; counterfactuals that keep that
# covariate raise, and those that replace it warn.

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from piano import Composer

N_CELLS = 200
N_GENES = 30
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
UNSEEN = 'not seen in training'  # Every unseen-value warning and error says this


# ---------------------------------------------------------------------------
# Helpers: small count matrices, models trained on them once, and queries with unseen values
# ---------------------------------------------------------------------------
def make_adata(seed=0, batches=('sc10Xv2', 'sc10Xv3'), donors=('d1', 'd2')):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({'batch': rng.choice(list(batches), N_CELLS), 'donor': rng.choice(list(donors), N_CELLS)},
                       index=[f'{seed}_{i}' for i in range(N_CELLS)])
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_query(column='batch', unseen=(('multiome', 30),)):
    # Query cells drawn like the training cells, with the first cells set to values not seen in training (in order)
    query = make_adata(seed=1)
    query.obs[column] = query.obs[column].astype(object)
    start = 0
    for value, n_cells in unseen:
        query.obs.iloc[start:start + n_cells, query.obs.columns.get_loc(column)] = value
        start += n_cells
    return query


def train(outdir, mode='piano'):
    # One epoch on the CPU, not deterministic (which needs CUBLAS_WORKSPACE_CONFIG on GPU nodes)
    covariates = (dict(categorical_covariate_keys=['batch', 'donor']) if mode == 'piano'
                  else dict(batch_key='batch', categorical_covariate_keys=['donor']))
    pianist = Composer(make_adata(), mode=mode, memory_mode='CPU', compile_model=False, n_top_genes=-1,
                       max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=outdir,
                       **covariates)
    pianist.run_pipeline()
    return pianist


@pytest.fixture(scope='module', params=['piano', 'scvi'])
def pianist(request, tmp_path_factory):
    """Trained with batches sc10Xv2/sc10Xv3 and donors d1/d2, in PIANO mode and in scVI mode (batch_key='batch')."""
    return train(str(tmp_path_factory.mktemp(f'unseen_categorical_{request.param}')), mode=request.param)


@pytest.fixture(scope='module')
def piano_pianist(tmp_path_factory):
    """PIANO mode only, for counterfactuals (scVI mode has none)."""
    return train(str(tmp_path_factory.mktemp('unseen_categorical_counterfactual')))


def unseen_warnings(function, *args, **kwargs):
    """Call function and return its result and the messages of the unseen-value warnings it gave."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = function(*args, **kwargs)
    # Unrelated warnings (anndata, torch deprecations, pin_memory) are ignored
    return result, [str(warning.message) for warning in caught if UNSEEN in str(warning.message)]


def raised_unseen_error(function, *args, **kwargs):
    """Call function, which must raise ValueError, and return its message; no unseen-value warning comes first."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError) as error:
            function(*args, **kwargs)
    assert not [warning for warning in caught if UNSEEN in str(warning.message)]
    return str(error.value)


def latents(pianist, query, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pianist.get_latent_representation(query, **kwargs)


def relabelled(query, column, value):
    # The same cells with every value of column replaced (value None: missing)
    query = query.copy()
    query.obs[column] = value
    return query


# ---------------------------------------------------------------------------
# get_latent_representation: unseen values warn, are encoded as unlabeled, and leave the latents unchanged
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('memory_mode', ['CPU', 'SparseCPU', pytest.param('GPU', marks=GPU)])
def test_latent_representation_of_an_unseen_batch_warns_and_is_returned(pianist, memory_mode):
    # E.g. a new dataset mapped onto a trained reference; before SCRUM-19 this raised a bare KeyError
    query = make_query()
    latent, messages = unseen_warnings(pianist.get_latent_representation, query, memory_mode=memory_mode)
    assert latent.shape[0] == N_CELLS and np.isfinite(latent).all()
    assert len(messages) == 1
    argument = 'batch_key' if pianist.mode == 'scvi' else 'categorical_covariate_keys'
    assert f"column 'batch' of {argument}" in messages[0] and "'multiome': 30 cells" in messages[0]
    assert 'latent representation does not use covariates' in messages[0]


def test_latents_do_not_depend_on_the_batch_label(pianist):
    # The encoder sees the counts alone: unseen, seen, unlabeled and missing labels give the same latents
    query = make_query()
    expected = latents(pianist, query)
    for value in ('sc10Xv2', 'sc10Xv3', 'Unknown', None, 'another_new_batch'):
        np.testing.assert_array_equal(latents(pianist, relabelled(query, 'batch', value)), expected)


def test_unseen_values_are_named_with_their_cell_counts(pianist):
    # So a typo stands out among real new batches; values sorted by str()
    query = make_query(unseen=(('multiome', 20), ('sc10xv3', 5)))
    _, messages = unseen_warnings(pianist.get_latent_representation, query)
    assert len(messages) == 1
    assert messages[0].index("'multiome': 20 cells") < messages[0].index("'sc10xv3': 5 cells")


def test_one_warning_per_column_with_unseen_values(pianist):
    query = make_query()
    query.obs['donor'] = query.obs['donor'].astype(object)
    query.obs.iloc[:7, query.obs.columns.get_loc('donor')] = 'd3'
    _, messages = unseen_warnings(pianist.get_latent_representation, query)
    assert sorted(message.split("'")[1] for message in messages) == ['batch', 'donor']
    assert any("'d3': 7 cells" in message for message in messages)


@pytest.mark.parametrize('query', ['training', 'seen'], ids=['training', 'seen_query'])
def test_seen_values_do_not_warn(pianist, query):
    adata = None if query == 'training' else make_adata(seed=2)
    _, messages = unseen_warnings(pianist.get_latent_representation, adata)
    assert messages == []


@pytest.mark.parametrize('missing', [np.nan, None], ids=['nan', 'none'])
def test_missing_values_are_not_reported_as_unseen(pianist, missing):
    # They are unlabeled, which training knows; their own warning (missing values) is unchanged
    query = make_query(unseen=((missing, 10),))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pianist.get_latent_representation(query)
    messages = [str(warning.message) for warning in caught]
    assert not [message for message in messages if UNSEEN in message]
    assert any('missing values' in message for message in messages)


def test_unseen_cells_are_exactly_those_encoded_as_unlabeled(pianist):
    # The check and the dataset's encoding agree: the reported cells, and only they, get zeros in every batch column
    query = make_query(unseen=(('multiome', 12),))
    unseen = pianist._get_unseen_categorical_values(query.obs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        covariates = pianist._get_adataset(query, 'CPU').aug_data.numpy()  # Covariates only; genes are stacked per batch
    n_batch_columns = 2  # sc10Xv2 and sc10Xv3, encoded first (the batch is the first categorical covariate)
    assert covariates.shape == (N_CELLS, 4)  # Two batches and two donors
    all_zero = (covariates[:, :n_batch_columns] == 0).all(axis=1)
    assert unseen == {'batch': {'multiome': 12}}
    assert all_zero[:12].all() and not all_zero[12:].any()


def test_query_without_the_batch_column_still_raises(pianist):
    query = make_adata(seed=1)
    del query.obs['batch']
    with pytest.raises(ValueError, match="'batch'"):
        pianist.get_latent_representation(query)


# ---------------------------------------------------------------------------
# get_counterfactual (PIANO mode): the decoder uses covariates, so kept unseen values raise and replaced ones warn
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('covariates', [None, {'donor': 'd1'}], ids=['reconstruction', 'other_column_replaced'])
def test_counterfactual_keeping_an_unseen_value_raises(piano_pianist, covariates):
    message = raised_unseen_error(piano_pianist.get_counterfactual, make_query(), covariates=covariates)
    assert "column 'batch' of categorical_covariate_keys" in message and "'multiome': 30 cells" in message
    assert 'decoded as unlabeled' in message and "covariates='marginal'" in message


def test_counterfactual_raises_only_for_kept_columns(piano_pianist):
    query = make_query()
    query.obs['donor'] = query.obs['donor'].astype(object)
    query.obs.iloc[:7, query.obs.columns.get_loc('donor')] = 'd3'
    message = raised_unseen_error(piano_pianist.get_counterfactual, query, covariates={'batch': 'sc10Xv2'})
    assert "column 'donor'" in message and "column 'batch'" not in message


@pytest.mark.parametrize('covariates', ['marginal', {'batch': 'sc10Xv2'}, {'batch': 'marginal'}],
                         ids=['marginal', 'value', 'column_marginal'])
def test_counterfactual_replacing_the_unseen_value_warns(piano_pianist, covariates):
    counts, messages = unseen_warnings(piano_pianist.get_counterfactual, make_query(), covariates=covariates)
    assert counts.shape == (N_CELLS, N_GENES) and np.isfinite(counts).all()
    assert len(messages) == 1 and 'replaced by the counterfactual values' in messages[0]
    assert 'latent' not in messages[0]
    # Replaced, the unseen value is not used at all: the same cells with a training label give the same counts
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = piano_pianist.get_counterfactual(relabelled(make_query(), 'batch', 'sc10Xv3'), covariates=covariates)
    np.testing.assert_array_equal(counts, expected)


def test_reconstruction_of_seen_values_does_not_warn(piano_pianist):
    _, messages = unseen_warnings(piano_pianist.get_counterfactual, make_adata(seed=2), covariates=None)
    assert messages == []


# ---------------------------------------------------------------------------
# The checks use what the dataset encodes, after the query's own checks
# ---------------------------------------------------------------------------
def test_unseen_categories_of_a_categorical_dtype_are_counted_only_if_used(pianist):
    # h5ad files load string columns as categoricals; an unused category is no cell's value
    query = make_query(unseen=(('multiome', 30),))
    query.obs['batch'] = pd.Categorical(query.obs['batch'], categories=['sc10Xv2', 'sc10Xv3', 'multiome', 'unused'])
    _, messages = unseen_warnings(pianist.get_latent_representation, query)
    assert len(messages) == 1 and "'multiome': 30 cells" in messages[0] and 'unused' not in messages[0]


@pytest.mark.parametrize('method', ['latent', 'counterfactual'])
def test_query_that_is_not_an_anndata_raises_the_type_error(pianist, piano_pianist, method):
    # The query checks run before the unseen-value checks, which would otherwise fail on .obs with an AttributeError
    function = (pianist.get_latent_representation if method == 'latent' else piano_pianist.get_counterfactual)
    with pytest.raises(TypeError, match='AnnData'):
        function('query.h5ad')


def test_query_that_fails_its_checks_does_not_warn_about_unseen_values(pianist):
    query = make_query()[:, :N_GENES - 1].copy()  # A training gene is missing
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match='gene'):
            pianist.get_latent_representation(query)
    assert not [warning for warning in caught if UNSEEN in str(warning.message)]


def write_backed_query(path, drop_columns=()):
    # The query (with its unseen batch) written as CSR and read back backed; drop_columns are left out of the file
    query = make_query()
    query.X = sp.csr_matrix(query.X)
    query.obs = query.obs.drop(columns=list(drop_columns))
    query.write_h5ad(path)
    return ad.read_h5ad(path, backed='r')


def test_backed_query_is_checked_on_the_obs_in_its_file(piano_pianist, tmp_path):
    # The backed dataset encodes the obs it reads from the file: relabelling only the in-memory obs does not hide the
    # unseen batch from the counterfactual check
    query = write_backed_query(tmp_path / 'query.h5ad')
    query.obs['batch'] = 'sc10Xv2'
    message = raised_unseen_error(piano_pianist.get_counterfactual, query, covariates=None, memory_mode='backed')
    assert "'multiome': 30 cells" in message


def test_backed_query_latents_warn_and_match_in_memory(pianist, tmp_path):
    query = write_backed_query(tmp_path / 'query.h5ad')
    latent, messages = unseen_warnings(pianist.get_latent_representation, query, memory_mode='backed')
    assert len(messages) == 1 and "'multiome': 30 cells" in messages[0]
    np.testing.assert_allclose(latent, latents(pianist, make_query()), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('memory_mode', ['CPU', 'backed'])
def test_counterfactual_raises_before_building_the_query_dataset(piano_pianist, monkeypatch, tmp_path, memory_mode):
    # A query refused for its unseen values is not first copied into a dataset (in GPU modes, onto the GPU)
    query = write_backed_query(tmp_path / 'query.h5ad') if memory_mode == 'backed' else make_query()
    built = []
    monkeypatch.setattr(piano_pianist, '_get_adataset', lambda *args, **kwargs: built.append(args))
    raised_unseen_error(piano_pianist.get_counterfactual, query, covariates=None, memory_mode=memory_mode)
    assert built == []


def test_backed_query_column_only_in_memory_raises(pianist, tmp_path):
    # Backed mode encodes the file's obs, so a column added only in memory cannot be used (it passes the query checks,
    # which read the in-memory obs)
    query = write_backed_query(tmp_path / 'query.h5ad', drop_columns=['donor'])
    query.obs['donor'] = 'd1'
    with pytest.raises(ValueError, match="backed mode reads obs from the file"):
        pianist.get_latent_representation(query, memory_mode='backed')
