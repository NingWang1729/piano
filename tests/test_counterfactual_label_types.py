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
DONORS = [10, 20, 30]
LEVELS = [0.5, 1.0]


# ---------------------------------------------------------------------------
# Helper: small count matrix whose categorical covariates are integers, floats and bools (not strings)
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({
        'donor': rng.choice(DONORS, N_CELLS),
        'level': rng.choice(LEVELS, N_CELLS),
        'treated': rng.choice([True, False], N_CELLS),
    }, index=[str(i) for i in range(N_CELLS)])
    adata = ad.AnnData(X=csr_matrix(X), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.fixture(scope='module')
def pianist(tmp_path_factory):
    # One epoch is enough: these tests only check how counterfactual values are matched to the training labels.
    # One memory mode is enough: counterfactual values are matched to labels from obs_encoding_dict on the host,
    # before any dataset or memory mode is used
    # Not deterministic, which needs CUBLAS_WORKSPACE_CONFIG on GPU nodes (review item 16)
    pianist = Composer(
        make_adata(), memory_mode='SparseCPU', compile_model=False, n_top_genes=-1,
        categorical_covariate_keys=['donor', 'level', 'treated'],
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False,
        outdir=str(tmp_path_factory.mktemp('counterfactual_label_types')),
    )
    pianist.run_pipeline()
    return pianist


def assert_same_encoding(pianist, covariates, string_covariates):
    # The counterfactual covariates given as numbers or bools are encoded exactly as their string labels
    mask, counterfactuals = pianist._encode_counterfactual_covariates(covariates)
    string_mask, string_counterfactuals = pianist._encode_counterfactual_covariates(string_covariates)
    np.testing.assert_array_equal(mask, string_mask)
    np.testing.assert_array_equal(counterfactuals, string_counterfactuals)


@pytest.mark.parametrize('covariates, string_covariates', [
    ({'donor': 20}, {'donor': '20'}),
    ({'donor': np.int64(20)}, {'donor': '20'}),
    ({'donor': [10, 30]}, {'donor': ['10', '30']}),
    ({'donor': [10, '30']}, {'donor': ['10', '30']}),
    ({'donor': [20, '20']}, {'donor': '20'}),  # one label given twice, as a number and a string, is one value
    ({'level': 1.0}, {'level': '1.0'}),
    ({'level': [0.5, 1.0]}, {'level': ['0.5', '1.0']}),
    ({'treated': True}, {'treated': 'True'}),
    ({'treated': np.bool_(False)}, {'treated': 'False'}),
    ({'treated': [True, False]}, {'treated': ['True', 'False']}),
    ({'donor': 10, 'level': 0.5, 'treated': False}, {'donor': '10', 'level': '0.5', 'treated': 'False'}),
])
def test_non_string_values_match_labels(pianist, covariates, string_covariates):
    # Labels are stored as strings (str(value)), so these used to raise "Invalid value '20' for covariate 'donor'"
    assert_same_encoding(pianist, covariates, string_covariates)


def test_non_string_values_in_get_counterfactual(pianist):
    # The public entry point accepts the same values
    counterfactual = pianist.get_counterfactual(covariates={'donor': 20, 'level': [0.5, 1.0], 'treated': True})
    assert counterfactual.shape == (N_CELLS, N_GENES)


def test_one_hot_encoding_of_integer_value(pianist):
    # The donor block comes first; donor 20 is the second of the sorted labels '10', '20', '30'
    mask, counterfactuals = pianist._encode_counterfactual_covariates({'donor': 20})
    np.testing.assert_array_equal(mask[:len(DONORS)], [0, 0, 0])
    np.testing.assert_array_equal(counterfactuals[:len(DONORS)], [0, 1, 0])


@pytest.mark.parametrize('value, printed', [(40, '40'), (40.0, '40.0'), ('40', '40'), ([10, 40], '40')])
def test_value_that_is_not_a_label_raises_with_valid_labels(pianist, value, printed):
    # Values that really are not labels still raise, showing the value as matched, and the message lists the valid labels
    with pytest.raises(ValueError, match=f"Invalid value '{printed}' for covariate 'donor'") as error:
        pianist._encode_counterfactual_covariates({'donor': value})
    for label in DONORS:
        assert f"'{label}'" in str(error.value), f"label {label} missing from: {error.value}"


def test_string_values_marginal_and_unlabeled_unchanged(pianist):
    # String labels, 'marginal' and the unlabeled value keep working on columns built from non-strings
    mask, counterfactuals = pianist._encode_counterfactual_covariates({'donor': '30'})
    np.testing.assert_array_equal(mask[:len(DONORS)], [0, 0, 0])
    np.testing.assert_array_equal(counterfactuals[:len(DONORS)], [0, 0, 1])

    counterfactuals = pianist._encode_counterfactual_covariates({'donor': 'marginal'})[1]
    np.testing.assert_array_equal(counterfactuals[:len(DONORS)], np.full(len(DONORS), 1 / len(DONORS), dtype=np.float32))

    counterfactuals = pianist._encode_counterfactual_covariates({'donor': 'Unknown'})[1]
    np.testing.assert_array_equal(counterfactuals[:len(DONORS)], [0, 0, 0])

    assert pianist.get_counterfactual(covariates='marginal').shape == (N_CELLS, N_GENES)
