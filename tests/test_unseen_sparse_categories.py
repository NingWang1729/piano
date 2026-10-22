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

import math
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
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
UNSEEN = 'not seen in training'  # Every unseen-category warning and error says this


# ---------------------------------------------------------------------------
# Helpers: small count matrix with sparse continuous covariates, models trained on it once, and the query checks
# ---------------------------------------------------------------------------
def make_adata(seed=0, drugs=('A', 'B'), perturbed_genes=None):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(N_CELLS, N_GENES)).astype(np.float32)
    obs = pd.DataFrame({
        'batch': rng.choice(['a', 'b'], N_CELLS),
        'age': rng.normal(size=N_CELLS),
        'drug': rng.choice(list(drugs), N_CELLS),
        'dose': rng.uniform(size=N_CELLS),
    }, index=[f'{seed}_{i}' for i in range(N_CELLS)])
    if perturbed_genes is not None:
        obs['gene'] = rng.choice(list(perturbed_genes), N_CELLS)
        obs['level'] = rng.uniform(size=N_CELLS)
    adata = ad.AnnData(X=csr_matrix(X), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def make_query(column='drug', unseen=(('C', 30),), **kwargs):
    # Query cells drawn like the training cells, with the first cells set to values not seen in training (in order)
    query = make_adata(seed=1, **kwargs)
    query.obs[column] = query.obs[column].astype(object)
    start = 0
    for value, n_cells in unseen:
        query.obs.iloc[start:start + n_cells, query.obs.columns.get_loc(column)] = value
        start += n_cells
    return query


def train(adatas, outdir, sparse_continuous_covariate_keys=(('drug', 'dose'),)):
    # One epoch in SparseCPU mode, the only mode with sparse continuous covariates; not deterministic, which needs
    # CUBLAS_WORKSPACE_CONFIG on GPU nodes and would stay on for later tests (review item 16)
    pianist = Composer(
        adatas, memory_mode='SparseCPU', compile_model=False, n_top_genes=-1,
        categorical_covariate_keys=['batch'], continuous_covariate_keys=['age'],
        sparse_continuous_covariate_keys=list(sparse_continuous_covariate_keys),
        max_epochs=1, batch_size=64, early_stopping=False, deterministic=False, outdir=outdir,
    )
    pianist.run_pipeline()
    return pianist


@pytest.fixture(scope='module')
def pianist(tmp_path_factory):
    """Trained with drug A or B."""
    return train(make_adata(), str(tmp_path_factory.mktemp('unseen_sparse')))


@pytest.fixture(scope='module')
def pianist_two_sparse(tmp_path_factory):
    """Trained with drug A or B and perturbed gene X or Y."""
    return train(
        make_adata(perturbed_genes=('X', 'Y')), str(tmp_path_factory.mktemp('unseen_two_sparse')),
        sparse_continuous_covariate_keys=[('drug', 'dose'), ('gene', 'level')],
    )


def unseen_warnings(function, *args, **kwargs):
    """Call function and return its result and the messages of the unseen-category warnings it gave."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = function(*args, **kwargs)
    # Unrelated warnings (anndata, torch deprecations, pin_memory) are ignored
    return result, [str(warning.message) for warning in caught if UNSEEN in str(warning.message)]


def raised_unseen_error(function, *args, **kwargs):
    """Call function, which must raise ValueError, and return its message; no unseen-category warning comes first."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError) as error:
            function(*args, **kwargs)
    assert not [warning for warning in caught if UNSEEN in str(warning.message)]
    return str(error.value)


def assert_names_unseen(message, column, counts):
    # The column, each unseen value with its number of cells, values sorted by str()
    assert UNSEEN in message
    assert f"'{column}'" in message
    for value, n_cells in counts:
        assert f'{value}: {n_cells} cells' in message
    positions = [message.index(f'{value}: ') for value, _ in counts]
    assert positions == sorted(positions)


def assert_latent_wording(message):
    # Latents do not use covariates, but reconstructions of these cells would treat them as untreated
    assert 'latent representation does not use covariates' in message
    assert 'untreated' in message


def assert_replaced_wording(message):
    # The counterfactual replaces the cells' original category, so the latent wording does not apply (design review)
    assert 'replaced' in message
    assert 'latent' not in message


# ---------------------------------------------------------------------------
# get_unseen_values: counts the values not in the known values and decides nothing
# ---------------------------------------------------------------------------
def test_get_unseen_values_counts_unseen_values():
    from piano.utils.covariates import get_unseen_values
    assert get_unseen_values(['A', 'C', 'C', 'B', 'D'], ['A', 'B']) == {'C': 2, 'D': 1}


def test_get_unseen_values_is_empty_when_all_seen():
    from piano.utils.covariates import get_unseen_values
    assert get_unseen_values(['A', 'B', 'A'], ['A', 'B']) == {}


def test_get_unseen_values_skips_unused_categories():
    # Categories of a categorical column that no cell has are not reported
    from piano.utils.covariates import get_unseen_values
    values = pd.Categorical(['A', 'C', 'C'], categories=['A', 'B', 'C', 'D'])
    assert get_unseen_values(values, ['A', 'B']) == {'C': 2}


@pytest.mark.parametrize('missing', [np.nan, None], ids=['nan', 'none'])
def test_get_unseen_values_counts_missing_values(missing):
    # NaN and None are counted together as one unseen value, NaN
    from piano.utils.covariates import get_unseen_values
    unseen = get_unseen_values(pd.Series(['A', missing, missing], dtype=object), ['A', 'B'])
    assert len(unseen) == 1
    (value, n_cells), = unseen.items()
    assert isinstance(value, float) and math.isnan(value) and n_cells == 2


def test_get_unseen_values_matches_dataset_encoding():
    # Unit test of the new helper: the same values count as seen as in the SparseCPU dataset, which looks values up in
    # a dict: int 1 and float 1.0 are the same key, str '1' is not (key normalisation is left to SCRUM-34)
    from piano.utils.covariates import get_unseen_values
    assert get_unseen_values([1.0, 2.0], [1, 2]) == {}
    assert get_unseen_values(['1', 2], [1, 2]) == {'1': 1}


# ---------------------------------------------------------------------------
# get_latent_representation warns, once per call and column, in every memory mode, and returns the latents
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('memory_mode', [
    'SparseCPU', 'CPU', 'backed', pytest.param('GPU', marks=GPU), pytest.param('SparseGPU', marks=GPU),
])
def test_latent_representation_warns_for_unseen_category(pianist, memory_mode, tmp_path):
    query = make_query()
    if memory_mode == 'backed':
        # Backed mode needs the query on disk; written to tmp_path only
        query.write_h5ad(tmp_path / 'query.h5ad')
        query = ad.read_h5ad(tmp_path / 'query.h5ad', backed='r')
    latent, messages = unseen_warnings(pianist.get_latent_representation, query, memory_mode=memory_mode)
    assert len(messages) == 1
    assert_names_unseen(messages[0], 'drug', [("'C'", 30)])
    assert_latent_wording(messages[0])
    assert latent.shape[0] == N_CELLS


def test_latent_representation_names_every_unseen_value(pianist):
    _, messages = unseen_warnings(pianist.get_latent_representation, make_query(unseen=(('D', 20), ('C', 30))))
    assert len(messages) == 1
    assert_names_unseen(messages[0], 'drug', [("'C'", 30), ("'D'", 20)])


@pytest.mark.parametrize('missing', [np.nan, None], ids=['nan', 'none'])
def test_latent_representation_reports_missing_category(pianist, missing):
    # Missing categories are zeroed like any unseen value, so they are reported too
    _, messages = unseen_warnings(pianist.get_latent_representation, make_query(unseen=((missing, 20),)))
    assert len(messages) == 1
    assert_names_unseen(messages[0], 'drug', [('NaN (missing)', 20)])


def test_latent_representation_warns_once_per_column(pianist_two_sparse):
    query = make_query(unseen=(('C', 30),), perturbed_genes=('X', 'Y'))
    query.obs['gene'] = query.obs['gene'].astype(object)
    query.obs.iloc[:40, query.obs.columns.get_loc('gene')] = 'Z'
    _, messages = unseen_warnings(pianist_two_sparse.get_latent_representation, query)
    assert len(messages) == 2
    drug_message, = [message for message in messages if "'drug'" in message]
    gene_message, = [message for message in messages if "'gene'" in message]
    assert_names_unseen(drug_message, 'drug', [("'C'", 30)])
    assert_names_unseen(gene_message, 'gene', [("'Z'", 40)])


def test_latent_representation_warns_on_every_call(pianist):
    # Each query may differ, so the warning is not deduplicated
    query = make_query()
    for _ in range(2):
        _, messages = unseen_warnings(pianist.get_latent_representation, query)
        assert len(messages) == 1


def test_str_query_category_of_int_training_category_is_unseen(tmp_path):
    # Training int 1 and query str '1' are different keys of the sparse encoding, so the query cells are zeroed and
    # reported as unseen; normalising keys (e.g. to str) is left to SCRUM-34
    pianist = train(make_adata(drugs=(1, 2)), str(tmp_path))
    query = make_adata(seed=1, drugs=(1, 2))
    n_ones = int((query.obs['drug'] == 1).sum())
    query.obs['drug'] = query.obs['drug'].astype(str)
    _, messages = unseen_warnings(pianist.get_latent_representation, query)
    assert len(messages) == 1
    assert_names_unseen(messages[0], 'drug', [("'1'", n_ones), ("'2'", N_CELLS - n_ones)])


@pytest.mark.parametrize('query', [False, True], ids=['training', 'seen_query'])
def test_latent_representation_seen_categories_do_not_warn(pianist, query):
    # Coverage (passes on master): no unseen-category warning for the training data or a query of seen categories
    adata = make_adata(seed=1) if query else None
    _, messages = unseen_warnings(pianist.get_latent_representation, adata)
    assert messages == []


def test_query_seen_across_training_adatas_does_not_warn(tmp_path):
    # Coverage (passes on master): categories are seen if any training AnnData has them
    pianist = train([make_adata(seed=0, drugs=('A',)), make_adata(seed=2, drugs=('B',))], str(tmp_path))
    _, messages = unseen_warnings(pianist.get_latent_representation, make_adata(seed=1))
    assert messages == []


def test_query_with_int_categories_as_floats_does_not_warn(tmp_path):
    # Coverage (passes on master): float 1.0 is the same key as int 1 in the sparse encoding, so it is seen
    pianist = train(make_adata(drugs=(1, 2)), str(tmp_path))
    query = make_adata(seed=1, drugs=(1, 2))
    query.obs['drug'] = query.obs['drug'].astype(float)
    _, messages = unseen_warnings(pianist.get_latent_representation, query)
    assert messages == []


def test_latent_representation_query_without_category_column(pianist):
    # Coverage (passes on master): a query without the category column is left to SCRUM-94, so CPU latents still work
    query = make_adata(seed=1)
    del query.obs['drug']
    latent, messages = unseen_warnings(pianist.get_latent_representation, query, memory_mode='CPU')
    assert messages == []
    assert latent.shape[0] == N_CELLS


# ---------------------------------------------------------------------------
# get_counterfactual raises when the cells' own unseen category would be decoded, and warns when it is replaced
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('covariates', [None, 'marginal', {'batch': 'a'}], ids=['reconstruction', 'marginal', 'batch'])
def test_counterfactual_using_unseen_category_raises(pianist, covariates):
    # 'marginal' averages the categorical covariates only and keeps each cell's own sparse covariate, so it raises too
    message = raised_unseen_error(pianist.get_counterfactual, make_query(), covariates=covariates)
    assert_names_unseen(message, 'drug', [("'C'", 30)])
    assert "{'drug': {}}" in message  # The fix: replace the covariate
    assert 'latent' not in message  # Wording that fits get_counterfactual (design review)


def test_counterfactual_default_covariates_raise(pianist):
    # Mostly repeats test_counterfactual_using_unseen_category_raises[marginal]; kept to check that the default call,
    # whose covariates are still 'marginal', now raises for such queries
    message = raised_unseen_error(pianist.get_counterfactual, make_query())
    assert_names_unseen(message, 'drug', [("'C'", 30)])


def test_counterfactual_raises_only_for_kept_columns(pianist_two_sparse):
    # Replacing the drug leaves the unseen perturbed gene in use, so the error names the gene only
    query = make_query(unseen=(('C', 30),), perturbed_genes=('X', 'Y'))
    query.obs['gene'] = query.obs['gene'].astype(object)
    query.obs.iloc[:40, query.obs.columns.get_loc('gene')] = 'Z'
    message = raised_unseen_error(pianist_two_sparse.get_counterfactual, query, covariates={'drug': {'A': 1.0}})
    assert_names_unseen(message, 'gene', [("'Z'", 40)])
    assert "'drug'" not in message

    # Replacing both only warns, once per column
    counts, messages = unseen_warnings(
        pianist_two_sparse.get_counterfactual, query, covariates={'drug': {'A': 1.0}, 'gene': {}},
    )
    assert len(messages) == 2
    assert counts.shape == (N_CELLS, N_GENES)


@pytest.mark.parametrize('replacement', [{'A': 1.0}, {}, None], ids=['value', 'empty', 'none'])
def test_counterfactual_replacing_unseen_category_warns(pianist, replacement):
    counts, messages = unseen_warnings(pianist.get_counterfactual, make_query(), covariates={'drug': replacement})
    assert len(messages) == 1
    assert_names_unseen(messages[0], 'drug', [("'C'", 30)])
    assert_replaced_wording(messages[0])
    assert counts.shape == (N_CELLS, N_GENES)


@pytest.mark.parametrize('query', [False, True], ids=['training', 'seen_query'])
def test_reconstruction_of_seen_categories_does_not_warn(pianist, query):
    # Coverage (passes on master): reconstructions of the training data or of seen categories neither raise nor warn
    adata = make_adata(seed=1) if query else None
    counts, messages = unseen_warnings(pianist.get_counterfactual, adata, covariates=None)
    assert messages == []
    assert counts.shape == (N_CELLS, N_GENES)


def test_counterfactual_in_other_modes_still_refuses_sparse_covariates(pianist):
    # Coverage (passes on master): outside SparseCPU the existing sparse covariate error comes first (#62)
    with pytest.raises(ValueError, match='only supported in SparseCPU memory mode'):
        pianist.get_counterfactual(make_query(), covariates=None, memory_mode='CPU')
