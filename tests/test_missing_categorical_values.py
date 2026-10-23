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

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 40
N_GENES = 20
SMALL_MODEL = {'n_hidden': 16, 'n_layers': 1, 'latent_size': 8}  # Fast to train; the encoding is what is tested
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
MEMORY_MODES = ['CPU', 'SparseCPU', 'backed', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)]

# One cycle of values per kind of missing value, repeated over the cells: each cycle has two labels and missing values
MISSING_KINDS = {
    'float_nan': np.array([1.0, 2.0, np.nan, 1.0]),
    'object_none': np.array(['a', 'b', None, 'a'], dtype=object),
    'object_none_and_nan': np.array(['a', 'b', None, np.nan], dtype=object),
    'string_na': pd.array(['a', 'b', pd.NA, 'a'], dtype='string'),
    'int64_na': pd.array([1, 2, pd.NA, 1], dtype='Int64'),
    'category_nan': pd.Categorical(['a', 'b', np.nan, 'a']),
}
LABELED = np.array(['a', 'b', 'a', 'b'], dtype=object)  # A column without missing values


# ---------------------------------------------------------------------------
# Helpers: small counts whose obs holds the given categorical columns, and a Composer that encodes them
# ---------------------------------------------------------------------------
def make_adata(columns, seed=0):
    """A small count matrix whose obs holds the given columns, each cycle of values repeated to N_CELLS values."""
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {key: values[np.arange(N_CELLS) % len(values)] for key, values in columns.items()},
        index=[f'cell_{i}' for i in range(N_CELLS)],
    )
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def read_backed(adata, path):
    """Write adata to path (in pytest's temporary folder) and read it back backed, so its counts stay on disk."""
    adata.write_h5ad(path)
    return ad.read_h5ad(path, backed='r')


def make_composer(adata, outdir, memory_mode='CPU', keys=('batch',), unlabeled='Unknown'):
    return Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=list(keys), n_top_genes=-1,
        unlabeled=unlabeled, max_epochs=1, batch_size=16, early_stopping=False, deterministic=False,
        outdir=str(outdir), **SMALL_MODEL,
    )


def make_encoded_composer(adata, outdir, memory_mode='CPU', unlabeled='Unknown'):
    # No training is needed: initialize_features sets the encodings that the datasets use.
    # The missing values warn (tested below), which is not what these tests check
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        composer = make_composer(adata, outdir, memory_mode, unlabeled=unlabeled)
        composer.initialize_features()
    return composer


def build_query_adataset(composer, adata, memory_mode):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return composer._get_adataset(adata, memory_mode)


def assert_encoding_has_only_labels(encoding, values, unlabeled='Unknown'):
    # The encoding holds the unlabeled value and the labels of the non-missing values: no 'nan', 'None' or '<NA>' key,
    # and no missing value itself
    assert set(encoding) == {unlabeled} | {str(value) for value in values.dropna().unique()}
    assert encoding[unlabeled] == -1


def assert_missing_cells_are_zeros(adataset, encoding):
    # 'batch' is the only covariate, so aug_data holds its one-hot columns; missing cells are all zeros, as unlabeled cells
    aug_data = adataset.aug_data.cpu().numpy()
    values = adataset.obs['batch']
    missing = values.isna().to_numpy()
    assert missing.any()
    assert aug_data.shape == (N_CELLS, len(encoding) - 1)
    assert (aug_data[missing] == 0).all()
    for row, value in zip(aug_data[~missing], values[~missing]):
        expected = np.zeros(len(encoding) - 1, dtype=np.float32)
        expected[encoding[str(value)]] = 1
        np.testing.assert_array_equal(row, expected)


def missing_value_warnings(caught, column):
    """The caught warnings about missing values in obs[column]."""
    return [w for w in caught if f".obs['{column}'] has" in str(w.message) and 'missing values' in str(w.message)]


# ---------------------------------------------------------------------------
# A column with no labeled values (only missing or unlabeled) across all training AnnDatas raises at Composer creation
# ---------------------------------------------------------------------------
NO_LABELS = {
    'unknown': np.array(['Unknown'], dtype=object),
    'float_nan': np.array([np.nan]),
    'object_none': np.array([None], dtype=object),
    'nan_and_unknown': np.array([np.nan, 'Unknown'], dtype=object),
    'string_na_and_unknown': pd.array([pd.NA, 'Unknown'], dtype='string'),
    'category_nan_and_unknown': pd.Categorical([np.nan, 'Unknown']),
}


def assert_raises_no_labels(make, column='batch'):
    # The message names the column and the argument to change; missing values may warn first
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(ValueError, match=f"'{column}'") as error:
            make()
    assert 'categorical_covariate_keys' in str(error.value), str(error.value)


@pytest.mark.parametrize('values', NO_LABELS.values(), ids=NO_LABELS.keys())
def test_column_without_labels_raises_when_composer_is_created(values, tmp_path):
    # Previously no error at creation: an all-'Unknown' column failed in run_pipeline with a bare AssertionError from
    # encode_categorical_covariates, and missing values trained as the categories 'nan', 'None' or '<NA>'
    adata = make_adata({'batch': values, 'donor': LABELED})
    assert_raises_no_labels(lambda: make_composer(adata, tmp_path, keys=('donor', 'batch')))


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_column_without_labels_raises_in_every_memory_mode(memory_mode, tmp_path):
    # Previously no error at creation; the check reads only obs, which backed AnnDatas also hold in memory
    adata = make_adata({'batch': NO_LABELS['category_nan_and_unknown']})
    if memory_mode == 'backed':
        adata = read_backed(adata, tmp_path / 'adata.h5ad')
    assert_raises_no_labels(lambda: make_composer(adata, tmp_path, memory_mode))


def test_column_without_labels_in_any_training_adata_raises(tmp_path):
    # Previously no error at creation. Labels are looked for across all training AnnDatas: neither of these has one
    first = make_adata({'batch': NO_LABELS['float_nan']}, seed=0)
    second = make_adata({'batch': NO_LABELS['unknown']}, seed=1)
    assert_raises_no_labels(lambda: make_composer([first, second], tmp_path))


def test_column_with_labels_in_another_training_adata_is_encoded(tmp_path):
    # Previously the missing cells of the first AnnData were encoded as a 'nan' category. A label in any training
    # AnnData is enough: the cells without one, in either AnnData, are encoded as unlabeled (all zeros)
    first = make_adata({'batch': NO_LABELS['nan_and_unknown']}, seed=0)
    second = make_adata({'batch': MISSING_KINDS['object_none']}, seed=1)
    composer = make_encoded_composer([first, second], tmp_path)
    assert set(composer.obs_encoding_dict['batch']) == {'Unknown', 'a', 'b'}

    aug_data = composer._get_training_adataset('CPU').datasets[0].aug_data.cpu().numpy()
    assert (aug_data == 0).all()


def test_custom_unlabeled_value_counts_as_no_label(tmp_path):
    # Previously no error at creation. With unlabeled='NA', 'NA' and missing values are no labels, but 'Unknown' is one
    no_labels = make_adata({'batch': np.array([np.nan, 'NA'], dtype=object)})
    assert_raises_no_labels(lambda: make_composer(no_labels, tmp_path, unlabeled='NA'))

    labeled = make_adata({'batch': np.array([np.nan, 'NA', 'Unknown'], dtype=object)})
    composer = make_encoded_composer(labeled, tmp_path, unlabeled='NA')
    assert set(composer.obs_encoding_dict['batch']) == {'NA', 'Unknown'}


def test_column_with_labels_in_one_training_adata_is_accepted(tmp_path):
    # Coverage: one labeled value in any training AnnData is enough, also when it is the default unlabeled value
    # 'Unknown' and the Composer's unlabeled value is another one
    first = make_adata({'batch': NO_LABELS['unknown']}, seed=0)
    second = make_adata({'batch': LABELED}, seed=1)
    make_composer([first, second], tmp_path)
    make_composer(first, tmp_path, unlabeled='NA')


# ---------------------------------------------------------------------------
# Missing values (NaN, None, pd.NA) are encoded as the unlabeled value, in training and query datasets
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kind', MISSING_KINDS)
@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_missing_values_encoded_as_unlabeled(memory_mode, kind, tmp_path):
    # Previously the missing values became the categories 'nan', 'None' or '<NA>' (two for None and NaN together),
    # each with its own one-hot column
    train, query = make_adata({'batch': MISSING_KINDS[kind]}, seed=0), make_adata({'batch': MISSING_KINDS[kind]}, seed=1)
    if memory_mode == 'backed':
        train, query = read_backed(train, tmp_path / 'train.h5ad'), read_backed(query, tmp_path / 'query.h5ad')
    composer = make_encoded_composer(train, tmp_path, memory_mode)
    encoding = composer.obs_encoding_dict['batch']
    assert_encoding_has_only_labels(encoding, train.obs['batch'])

    assert_missing_cells_are_zeros(composer._get_training_adataset(memory_mode), encoding)
    assert_missing_cells_are_zeros(build_query_adataset(composer, query, memory_mode), encoding)


@pytest.mark.parametrize('kind', ['float_nan', 'object_none', 'string_na', 'category_nan'])
def test_query_missing_values_without_missing_training_values(kind, tmp_path):
    # Previously a bare KeyError ('nan', 'None' or '<NA>') when building the query dataset: training had no such category.
    # One memory mode is enough: every dataset looks cells up the same way (AnnDataset._get_categorical_augmented_matrix)
    labels = np.array([1.0, 2.0]) if kind == 'float_nan' else LABELED
    composer = make_encoded_composer(make_adata({'batch': labels}), tmp_path)
    encoding = composer.obs_encoding_dict['batch']
    assert_missing_cells_are_zeros(
        build_query_adataset(composer, make_adata({'batch': MISSING_KINDS[kind]}, seed=1), 'CPU'), encoding,
    )


def test_missing_values_encoded_as_custom_unlabeled_value(tmp_path):
    # Previously NaN was a 'nan' category. With unlabeled='NA', missing cells are encoded as 'NA' cells are: all zeros
    composer = make_encoded_composer(
        make_adata({'batch': np.array(['a', 'NA', np.nan, 'b'], dtype=object)}), tmp_path, unlabeled='NA',
    )
    encoding = composer.obs_encoding_dict['batch']
    assert set(encoding) == {'NA', 'a', 'b'}

    aug_data = composer._get_training_adataset('CPU').aug_data.cpu().numpy()
    values = composer.adata[0].obs['batch']
    assert (aug_data[values.isna().to_numpy()] == 0).all()
    assert (aug_data[(values == 'NA').to_numpy()] == 0).all()


def test_literal_missing_value_strings_stay_categories(tmp_path):
    # Coverage: only true missing values are unlabeled; the strings 'nan', 'None' and '<NA>' are ordinary labels
    composer = make_encoded_composer(
        make_adata({'batch': np.array(['nan', 'None', '<NA>', np.nan], dtype=object)}), tmp_path,
    )
    assert set(composer.obs_encoding_dict['batch']) == {'Unknown', 'nan', 'None', '<NA>'}


# ---------------------------------------------------------------------------
# Missing values warn once per column, with the number of cells: training data at Composer creation, query data in
# get_latent_representation and get_counterfactual. The warnings read only obs, so one memory mode is enough
# ---------------------------------------------------------------------------
def assert_warns_once(caught, column, index, n_missing):
    # One warning for the column, with its number of missing cells and the fix, pointing at the user's call (this file)
    found = missing_value_warnings(caught, column)
    assert len(found) == 1, [str(w.message) for w in caught]
    message = str(found[0].message)
    assert f"adata {index}.obs['{column}'] has {n_missing} missing values" in message, message
    assert "'Unknown'" in message, message
    assert found[0].filename == __file__


def test_missing_training_values_warn_once_per_column_when_composer_is_created(tmp_path):
    # Previously no warning. 'batch' has 10 missing cells in adata 0 and 'donor' 20 in adata 1
    first = make_adata({'batch': MISSING_KINDS['float_nan'], 'donor': LABELED}, seed=0)
    second = make_adata({'batch': LABELED, 'donor': MISSING_KINDS['object_none_and_nan']}, seed=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        make_composer([first, second], tmp_path, keys=('batch', 'donor'))
    assert_warns_once(caught, 'batch', 0, 10)
    assert_warns_once(caught, 'donor', 1, 20)


def test_columns_without_missing_values_do_not_warn(tmp_path):
    # Coverage: cells holding the unlabeled value itself are not missing values
    adata = make_adata({'batch': LABELED, 'donor': np.array(['a', 'Unknown'], dtype=object)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        make_composer(adata, tmp_path, keys=('batch', 'donor'))
    assert not missing_value_warnings(caught, 'batch') + missing_value_warnings(caught, 'donor')


@pytest.fixture(scope='module')
def trained_pianist(tmp_path_factory):
    # One CPU epoch on training data with missing values (cheap end-to-end check); the model and logs go to pytest's
    # temporary folder, not next to the data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pianist = make_composer(
            make_adata({'batch': MISSING_KINDS['object_none'], 'donor': LABELED}), tmp_path_factory.mktemp('trained'),
            keys=('batch', 'donor'),
        )
        pianist.run_pipeline()
    return pianist


def test_training_with_missing_values_runs_end_to_end(trained_pianist):
    # Previously trained with a 'None' category; now the missing cells are unlabeled
    assert set(trained_pianist.obs_encoding_dict['batch']) == {'Unknown', 'a', 'b'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        query = make_adata({'batch': MISSING_KINDS['object_none_and_nan'], 'donor': LABELED}, seed=2)
        assert np.isfinite(trained_pianist.get_latent_representation()).all()
        assert np.isfinite(trained_pianist.get_latent_representation(query)).all()
        assert np.isfinite(trained_pianist.get_counterfactual(query, covariates=None)).all()


QUERY_METHODS = {
    'latent': lambda pianist, query: pianist.get_latent_representation(adata=query),
    'counterfactual': lambda pianist, query: pianist.get_counterfactual(adata=query, covariates='marginal'),
}


@pytest.mark.parametrize('method', QUERY_METHODS.values(), ids=QUERY_METHODS.keys())
def test_missing_query_values_warn_once_per_column_per_call(trained_pianist, method):
    # Previously no warning. 'batch' has 10 missing cells and 'donor' none; each call warns again
    query = make_adata({'batch': MISSING_KINDS['object_none'], 'donor': LABELED}, seed=3)
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            method(trained_pianist, query)
        assert_warns_once(caught, 'batch', 'query', 10)
        assert not missing_value_warnings(caught, 'donor')


@pytest.mark.parametrize('method', QUERY_METHODS.values(), ids=QUERY_METHODS.keys())
def test_query_without_missing_values_does_not_warn(trained_pianist, method):
    # Coverage: unlabeled query cells are not missing values
    query = make_adata({'batch': np.array(['a', 'Unknown'], dtype=object), 'donor': LABELED}, seed=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        method(trained_pianist, query)
    assert not missing_value_warnings(caught, 'batch') + missing_value_warnings(caught, 'donor')
