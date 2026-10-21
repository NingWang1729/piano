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
# Helper: small counts with a batch and a sparse continuous ('drug', 'dose') covariate
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], N_CELLS)),
        'drug': rng.choice(['A', 'B'], N_CELLS),
        'dose': rng.uniform(size=N_CELLS),
    }, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_composer(adata, outdir):
    # SparseCPU is the only memory mode with sparse continuous covariates: the other modes raise at creation (#62)
    # before these checks run. Not deterministic, which needs CUBLAS_WORKSPACE_CONFIG on GPU nodes (review item 16)
    return Composer(
        adata, memory_mode='SparseCPU', compile_model=False, categorical_covariate_keys=['batch'],
        sparse_continuous_covariate_keys=[('drug', 'dose')], n_top_genes=-1, max_epochs=1, batch_size=64,
        early_stopping=False, deterministic=False, outdir=str(outdir),
    )


@pytest.mark.parametrize('column', ['drug', 'dose'], ids=['category', 'value'])
def test_missing_sparse_continuous_covariate_column_raises_when_composer_is_created(column, tmp_path):
    # Previously creation passed and run_pipeline failed with a bare KeyError, which did not say which AnnData or
    # argument named the column
    second = make_adata(seed=1)
    del second.obs[column]
    message = rf"adata 1\.obs has no column '{column}', which sparse_continuous_covariate_keys names"
    with pytest.raises(ValueError, match=message):
        make_composer([make_adata(), second], tmp_path)


@pytest.mark.parametrize('invalid_value', [np.nan, np.inf, -np.inf, 1e39], ids=['nan', 'inf', '-inf', 'float32_overflow'])
def test_invalid_sparse_continuous_value_raises_when_composer_is_created(invalid_value, tmp_path):
    # Previously creation passed and training failed in the first epoch with torch's
    # "Expected parameter loc ... Normal ... Real()"
    adata = make_adata()
    adata.obs.loc['0', 'dose'] = invalid_value
    message = r"adata 0\.obs\['dose'\] has [01] NaN and [01] infinite values, but continuous covariates must be finite"
    with pytest.raises(ValueError, match=message):
        make_composer(adata, tmp_path)


def test_missing_value_in_nullable_value_column_raises(tmp_path):
    adata = make_adata()
    adata.obs['dose'] = adata.obs['dose'].astype('Float64')
    adata.obs.loc['0', 'dose'] = pd.NA
    with pytest.raises(ValueError, match=r"adata 0\.obs\['dose'\] has 1 NaN and 0 infinite values"):
        make_composer(adata, tmp_path)


def test_error_names_the_adata_with_invalid_values(tmp_path):
    second = make_adata(seed=1)
    second.obs.loc['5', 'dose'] = np.nan
    with pytest.raises(ValueError, match=r"adata 1\.obs\['dose'\] has 1 NaN"):
        make_composer([make_adata(), second], tmp_path)


def test_valid_sparse_continuous_covariates_train(tmp_path):
    # Coverage: finite values pass the checks and train for one epoch
    pianist = make_composer(make_adata(), tmp_path)
    pianist.run_pipeline()
    assert np.isfinite(pianist.get_latent_representation()).all()


NON_NUMERIC_DOSES = {
    'object_strings': lambda dose: dose.astype(object).where(dose.index != '0', 'abc'),
    'object_number_like_strings': lambda dose: dose.astype(str).astype(object),
    'object_floats': lambda dose: dose.astype(object),
    'string_dtype': lambda dose: dose.astype(str).astype('string'),
    'numeric_categorical': lambda dose: (dose * 10).round().astype('category'),
    'bool': lambda dose: dose > 0.5,
    'nullable_boolean': lambda dose: (dose > 0.5).astype('boolean'),
    'complex': lambda dose: dose.astype(np.complex64),
}


@pytest.mark.parametrize('make_dose', NON_NUMERIC_DOSES.values(), ids=NON_NUMERIC_DOSES.keys())
def test_non_numeric_sparse_continuous_value_dtype_raises_when_composer_is_created(make_dose, tmp_path):
    # Previously 'abc' failed in training with numpy's "could not convert string to float: 'abc'", naming no column, and
    # number-like strings, string dtype, bools and numeric categories trained without error
    adata = make_adata()
    adata.obs['dose'] = make_dose(adata.obs['dose'])
    with pytest.raises(
        ValueError, match=r"adata 0\.obs\['dose'\] has dtype .*, but continuous covariates must be numbers \(float or int\)",
    ) as error:
        make_composer(adata, tmp_path)
    # A value column cannot become a categorical covariate, so the advice must not suggest it
    assert 'categorical_covariate_keys' not in str(error.value)


def test_dtype_is_checked_before_values(tmp_path):
    # A categorical value column that also holds a NaN gets the dtype error, which is the one to fix first
    adata = make_adata()
    adata.obs['dose'] = (adata.obs['dose'] * 10).round().astype('category')
    adata.obs.loc['0', 'dose'] = np.nan
    with pytest.raises(ValueError, match=r"has dtype .*, but continuous covariates must be numbers"):
        make_composer(adata, tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int64', 'uint8', 'Float64', 'Int64'])
def test_numeric_sparse_continuous_value_dtypes_are_accepted(dtype, tmp_path):
    # Coverage: floats in any precision, signed and unsigned ints, and nullable floats and ints without missing values
    # pass the checks and train for one epoch
    adata = make_adata()
    adata.obs['dose'] = (adata.obs['dose'] * 10).round().astype(dtype)  # Non-negative whole numbers for uint8 and Int64
    pianist = make_composer(adata, tmp_path)
    pianist.run_pipeline()
    assert np.isfinite(pianist.get_latent_representation()).all()
