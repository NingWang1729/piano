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

import numpy as np
import pandas as pd
import pytest

from piano.utils.covariates import encode_categorical_covariates

# Columns of each kind a categorical covariate can be; the labels are str() of their values (see covariates.py)
COLUMNS = {
    'str': ['b', 'a', 'c', 'a', 'Unknown'],
    'int': [3, 1, 2, 1, 3],
    'bool': [True, False, True, True, False],
    'categorical': pd.Categorical(['b', 'a', 'b', 'a', 'a']),
}
# The labels each column encodes to, in code order (sorted as strings)
EXPECTED_LABELS = {
    'str': ['a', 'b', 'c'],
    'int': ['1', '2', '3'],
    'bool': ['False', 'True'],
    'categorical': ['a', 'b'],
}


@pytest.mark.parametrize('kind', list(COLUMNS))
def test_encoding_gives_string_labels_in_sorted_order_without_warnings(kind):
    # FutureWarning as an error: pandas 2 warns that factorize on a list "will raise in a future version", and pandas 3
    # raises TypeError, which failed every Composer with a categorical covariate
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        counterfactual, encoding, decoding = encode_categorical_covariates(pd.DataFrame({'key': COLUMNS[kind]}), ['key'])

    labels = EXPECTED_LABELS[kind]
    assert encoding['key'] == {'Unknown': -1} | {label: code for code, label in enumerate(labels)}
    assert decoding['key'] == {-1: 'Unknown'} | dict(enumerate(labels))
    # Labels are Python strings and codes integers, as before, so lookups and decoded counterfactual labels are unchanged
    assert all(type(label) is str for label in encoding['key'])
    assert all(isinstance(code, (int, np.integer)) for code in encoding['key'].values())
    assert np.allclose(counterfactual, [1 / len(labels)] * len(labels))
