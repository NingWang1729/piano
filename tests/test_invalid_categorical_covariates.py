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


import re
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50
SMALL_MODEL = {'n_hidden': 16, 'n_layers': 1, 'latent_size': 8}  # Fast to train; the covariate labels are what is tested
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
MEMORY_MODES = ['CPU', 'SparseCPU', 'backed', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU)]


# ---------------------------------------------------------------------------
# Helpers: small counts with categorical 'batch' and 'donor' covariates
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS)),
        'donor': rng.choice(['d1', 'd2', 'd3'], N_CELLS),
    }, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(adata, outdir):
    return Composer(
        adata, memory_mode='SparseCPU', compile_model=False, categorical_covariate_keys=['batch', 'donor'],
        n_top_genes=-1, max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir),
    )


QUERY_METHODS = {
    'latent': lambda pianist, query: pianist.get_latent_representation(adata=query),
    'counterfactual': lambda pianist, query: pianist.get_counterfactual(adata=query, covariates='marginal'),
}


# ---------------------------------------------------------------------------
# Missing columns
# ---------------------------------------------------------------------------
def test_missing_categorical_covariate_column_raises_when_composer_is_created(tmp_path):
    # Previously no error at creation; training then failed with a bare AssertionError from encode_categorical_covariates
    second = make_adata(seed=1)
    del second.obs['donor']
    with pytest.raises(ValueError, match=r"adata 1\.obs has no column 'donor', which categorical_covariate_keys names"):
        make_composer([make_adata(), second], tmp_path)


@pytest.fixture(scope='module')
def trained_pianist(tmp_path_factory):
    # One epoch in SparseCPU mode; the model and logs go to a temporary directory from pytest, not next to the data
    pianist = make_composer(make_adata(), tmp_path_factory.mktemp('query'))
    pianist.run_pipeline()
    return pianist


@pytest.mark.parametrize('method', QUERY_METHODS.values(), ids=QUERY_METHODS.keys())
def test_missing_query_categorical_covariate_column_raises(trained_pianist, method):
    # Previously a bare KeyError from building the query dataset
    query = make_adata(seed=2)
    del query.obs['donor']
    with pytest.raises(ValueError, match=r"adata query\.obs has no column 'donor', which categorical_covariate_keys names"):
        method(trained_pianist, query)


# ---------------------------------------------------------------------------
# Labels: each cell is looked up by the value Series.unique() returns, as the encoding's labels are str() of those values
# ---------------------------------------------------------------------------
def make_covariate_adata(columns, seed=0):
    """A small count matrix whose obs holds the given columns, each repeated to N_CELLS values."""
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {key: values[np.arange(N_CELLS) % len(values)] for key, values in columns.items()},
        index=[str(i) for i in range(N_CELLS)],
    )
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def read_backed(adata, path):
    """Write adata to path and read it back backed, so its counts stay on disk."""
    adata.write_h5ad(path)
    return ad.read_h5ad(path, backed='r')


def train_covariate_composer(adata, memory_mode, outdir, categorical_covariate_keys):
    composer = Composer(
        adata, memory_mode=memory_mode, compile_model=False, categorical_covariate_keys=categorical_covariate_keys,
        n_top_genes=-1, max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir),
        **SMALL_MODEL,
    )
    composer.run_pipeline()
    return composer


@pytest.mark.parametrize('memory_mode', MEMORY_MODES)
def test_float32_categorical_covariate_trains_and_queries(memory_mode, tmp_path):
    # Previously KeyError '0.10000000149011612': cells were labelled by str() of the Python float they iterate as,
    # while the encoding uses str() of the float32 values from unique(), '0.1' and '0.2'
    columns = {'cov': np.array([0.1, 0.2], dtype=np.float32)}
    train, query = make_covariate_adata(columns, seed=0), make_covariate_adata(columns, seed=1)
    if memory_mode == 'backed':
        # Files go to the temporary directory from pytest
        train, query = read_backed(train, tmp_path / 'train.h5ad'), read_backed(query, tmp_path / 'query.h5ad')
    pianist = train_covariate_composer(train, memory_mode, tmp_path, ['cov'])
    assert np.isfinite(pianist.get_latent_representation()).all()
    assert np.isfinite(pianist.get_counterfactual(covariates={'cov': '0.1'})).all()
    assert np.isfinite(pianist.get_latent_representation(query)).all()
    assert np.isfinite(pianist.get_counterfactual(query, covariates={'cov': '0.2'})).all()


def test_equal_int_and_float_values_share_a_label(tmp_path):
    # Previously KeyError '1.0': unique() returns 1 for both 1 and 1.0, so the encoding has no label '1.0'.
    # Checks on columns that mix Python types come in a separate PR.
    columns = {'cov': np.array([1, 1.0, 2], dtype=object)}
    pianist = train_covariate_composer(make_covariate_adata(columns), 'CPU', tmp_path, ['cov'])
    assert np.isfinite(pianist.get_counterfactual(covariates={'cov': '1'})).all()
    with pytest.raises(ValueError, match=r"Invalid value '1\.0' for covariate 'cov'"):
        pianist.get_counterfactual(covariates={'cov': '1.0'})


def test_labels_that_worked_before_are_unchanged(tmp_path):
    # Coverage: these columns already trained on master; their labels, including those of missing values, stay the same
    columns = {
        'none_nan': np.array([None, np.nan, 'x'], dtype=object),
        'int_na': pd.array([1, pd.NA, 2], dtype='Int64'),
        'categorical_float32': pd.Categorical(np.array([0.1, 0.2], dtype=np.float32)),
    }
    pianist = train_covariate_composer(make_covariate_adata(columns), 'CPU', tmp_path, list(columns))
    for key, label in [
        ('none_nan', 'None'), ('none_nan', 'nan'), ('none_nan', 'x'), ('int_na', '<NA>'), ('int_na', '1'),
        ('categorical_float32', '0.10000000149011612'),
    ]:
        assert np.isfinite(pianist.get_counterfactual(covariates={key: label})).all()


# ---------------------------------------------------------------------------
# Float columns: values that are not whole numbers usually measure a quantity, so they warn but are still used
# ---------------------------------------------------------------------------
NON_WHOLE_FLOAT_WARNING = r"adata {}\.obs\['cov'\] .*2 distinct.*continuous_covariate_keys"


def make_float_composer(adata, outdir):
    return Composer(
        adata, memory_mode='CPU', compile_model=False, categorical_covariate_keys=['cov'], n_top_genes=-1, max_epochs=1,
        batch_size=64, early_stopping=False, deterministic=False, outdir=str(outdir), **SMALL_MODEL,
    )


@pytest.mark.parametrize('values', [
    np.array([0.1, 0.2]), np.array([0.1, 0.2], dtype=np.float32), pd.Categorical([0.1, 0.2]),
], ids=['float64', 'float32', 'categorical_float64'])
def test_non_whole_float_categorical_covariate_warns_when_composer_is_created(values, tmp_path):
    # Previously no warning
    adata = make_covariate_adata({'cov': values})
    with pytest.warns(UserWarning, match=NON_WHOLE_FLOAT_WARNING.format('0')):
        make_float_composer(adata, tmp_path)


def test_non_whole_float_query_categorical_covariate_warns(tmp_path):
    # Previously no warning; the model is trained on the same values, so the query is only warned about
    columns = {'cov': np.array([0.1, 0.2])}
    with pytest.warns(UserWarning, match=NON_WHOLE_FLOAT_WARNING.format('0')):
        pianist = make_float_composer(make_covariate_adata(columns, seed=0), tmp_path)
    pianist.run_pipeline()
    with pytest.warns(UserWarning, match=NON_WHOLE_FLOAT_WARNING.format('query')):
        pianist.get_latent_representation(make_covariate_adata(columns, seed=1))


@pytest.mark.parametrize('values', [
    np.array([1.0, 2.0]), np.array([1.0, 2.0], dtype=np.float32), np.array([1.0, np.nan, 2.0]),
], ids=['float64', 'float32', 'float64_with_nan'])
def test_whole_float_categorical_covariate_does_not_warn(values, tmp_path):
    # Coverage: whole-number floats are often integer labels stored as floats; missing values are not checked
    # Only warnings naming the 'cov' column count: the Composer may warn about other things, e.g. sharing the counts (#76)
    adata = make_covariate_adata({'cov': values})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        make_float_composer(adata, tmp_path)
    assert not [w for w in caught if re.search(r"\.obs\['cov'\]", str(w.message))]
