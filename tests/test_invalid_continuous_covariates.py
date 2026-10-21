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
from scipy.sparse import csr_matrix

from piano import Composer

N_CELLS = 200
N_GENES = 50


# ---------------------------------------------------------------------------
# Helper: small counts with a batch and a continuous 'age' covariate
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS)),
        'age': rng.normal(0.0, 1.0, N_CELLS),
    }, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(adata, outdir):
    return Composer(
        adata, memory_mode='SparseCPU', compile_model=False, categorical_covariate_keys=['batch'],
        continuous_covariate_keys=['age'], n_top_genes=-1, max_epochs=1, batch_size=64, early_stopping=False,
        deterministic=False, outdir=str(outdir),
    )


@pytest.mark.parametrize('invalid_value', [np.nan, np.inf, -np.inf, 1e39], ids=['nan', 'inf', '-inf', 'float32_overflow'])
def test_invalid_continuous_covariate_raises_when_composer_is_created(invalid_value, tmp_path):
    # Previously one NaN gave NaN losses from the first epoch, and training failed with an error from torch or with
    # "Expected state_dict to be dict-like, got <class 'NoneType'>"
    adata = make_adata()
    adata.obs.loc['0', 'age'] = invalid_value
    message = r"adata 0\.obs\['age'\] has [01] NaN and [01] infinite values, but continuous covariates must be finite"
    with pytest.raises(ValueError, match=message):
        make_composer(adata, tmp_path)


def test_missing_value_in_nullable_column_raises(tmp_path):
    adata = make_adata()
    adata.obs['age'] = adata.obs['age'].astype('Float64')
    adata.obs.loc['0', 'age'] = pd.NA
    with pytest.raises(ValueError, match=r"adata 0\.obs\['age'\] has 1 NaN and 0 infinite values"):
        make_composer(adata, tmp_path)


def test_error_names_the_adata_with_invalid_values(tmp_path):
    second = make_adata(seed=1)
    second.obs.loc['5', 'age'] = np.nan
    with pytest.raises(ValueError, match=r"adata 1\.obs\['age'\] has 1 NaN"):
        make_composer([make_adata(), second], tmp_path)


def test_valid_continuous_covariates_train(tmp_path):
    # Finite continuous covariates pass the check and train for one epoch
    pianist = make_composer(make_adata(), tmp_path)
    pianist.run_pipeline()
    assert np.isfinite(pianist.get_latent_representation()).all()


def test_missing_continuous_covariate_column_raises_when_composer_is_created(tmp_path):
    # Previously a bare KeyError from pandas, which did not say which AnnData or argument named the column
    second = make_adata(seed=1)
    del second.obs['age']
    with pytest.raises(ValueError, match=r"adata 1\.obs has no column 'age', which continuous_covariate_keys names"):
        make_composer([make_adata(), second], tmp_path)


NON_NUMERIC_AGES = {
    'object_strings': lambda age: age.astype(object).where(age.index != '0', 'abc'),
    'object_number_like_strings': lambda age: age.astype(str).astype(object),
    'object_floats': lambda age: age.astype(object),
    'string_dtype': lambda age: age.astype(str).astype('string'),
    'numeric_categorical': lambda age: age.round().astype('category'),
    'bool': lambda age: age > 0,
    'nullable_boolean': lambda age: (age > 0).astype('boolean'),
    'complex': lambda age: age.astype(np.complex64),
}


@pytest.mark.parametrize('make_age', NON_NUMERIC_AGES.values(), ids=NON_NUMERIC_AGES.keys())
def test_non_numeric_continuous_covariate_dtype_raises_when_composer_is_created(make_age, tmp_path):
    # Previously strings failed with numpy's "could not convert string to float: 'abc'", and number-like strings,
    # categories and bools were accepted at creation, though object columns then failed in torch when training
    adata = make_adata()
    adata.obs['age'] = make_age(adata.obs['age'])
    with pytest.raises(
        ValueError, match=r"adata 0\.obs\['age'\] has dtype .*, but continuous covariates must be numbers \(float or int\)",
    ) as error:
        make_composer(adata, tmp_path)
    if pd.api.types.is_bool_dtype(adata.obs['age']):
        assert 'categorical_covariate_keys' in str(error.value)


def test_dtype_is_checked_before_values(tmp_path):
    # A categorical column that also holds a NaN gets the dtype error, which is the one to fix first
    adata = make_adata()
    adata.obs['age'] = adata.obs['age'].round().astype('category')
    adata.obs.loc['0', 'age'] = np.nan
    with pytest.raises(ValueError, match=r"has dtype .*, but continuous covariates must be numbers"):
        make_composer(adata, tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int64', 'uint8', 'Float64', 'Int64'])
def test_numeric_continuous_covariate_dtypes_are_accepted(dtype, tmp_path):
    # Coverage: floats in any precision, signed and unsigned ints, and nullable floats and ints without missing values
    # pass the checks and train for one epoch (training reads the column into a float32 tensor)
    adata = make_adata()
    adata.obs['age'] = (adata.obs['age'].abs() * 10).round().astype(dtype)  # Non-negative whole numbers for uint8 and Int64
    pianist = make_composer(adata, tmp_path)
    pianist.run_pipeline()
    assert np.isfinite(pianist.get_latent_representation()).all()


def test_bool_column_is_accepted_as_categorical_covariate(tmp_path):
    # Coverage: the error message's advice for bool columns works
    adata = make_adata()
    adata.obs['old'] = adata.obs['age'] > 0
    Composer(
        adata, memory_mode='SparseCPU', compile_model=False, categorical_covariate_keys=['batch', 'old'],
        continuous_covariate_keys=['age'], n_top_genes=-1, max_epochs=1, batch_size=64, early_stopping=False,
        deterministic=False, outdir=str(tmp_path),
    )
