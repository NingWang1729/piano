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

LABELED = np.array(['a', 'b', 'a', 'b'], dtype=object)  # A column without missing values


# ---------------------------------------------------------------------------
# Helpers: small counts whose obs holds the given categorical columns, and a Composer that uses them
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


def test_custom_unlabeled_value_counts_as_no_label(tmp_path):
    # Previously no error at creation. With unlabeled='NA', 'NA' and missing values are no labels
    no_labels = make_adata({'batch': np.array([np.nan, 'NA'], dtype=object)})
    assert_raises_no_labels(lambda: make_composer(no_labels, tmp_path, unlabeled='NA'))



def test_column_with_labels_in_one_training_adata_is_accepted(tmp_path):
    # Coverage: one labeled value in any training AnnData is enough, also when it is the default unlabeled value
    # 'Unknown' and the Composer's unlabeled value is another one
    first = make_adata({'batch': NO_LABELS['unknown']}, seed=0)
    second = make_adata({'batch': LABELED}, seed=1)
    make_composer([first, second], tmp_path)
    make_composer(first, tmp_path, unlabeled='NA')
