from typing import Iterable

import numpy as np
import pandas as pd


def encode_categorical_covariates(obs_list: pd.DataFrame | Iterable[pd.DataFrame], categorical_covariate_keys, unlabeled: str = 'Unknown'):
    """
    This function encodes categorical covariates from category strings into integer encodings.
    For a given column, a mapping is retrieved between the category and an integer encoding.
    This integer encoding is later used by the decoder and GLM to create a one-hot encoding of the covariate.
    Example:
        batch: ['A', 'B', 'C'] -> {'A': 0, 'B': 1, 'C': 2, 'Unknown': -1}
        Equivalent one hot encoding for the decoder and GLM (covariate used in covariate matrix): A: [1, 0, 0]
    Missing values (NaN, None, pd.NA) are not categories: their cells are encoded as unlabeled (categorical_labels).
    """
    if not isinstance(obs_list, list):
        obs_list = [obs_list]
    counterfactual_categorical_covariates_list = []
    obs_encoding_dict = {}
    obs_decoding_dict = {}
    for categorical_covariate_key in categorical_covariate_keys:
        # Retrieve set of labels across all datasets
        labels = set()
        for idx, obs in enumerate(obs_list):
            assert categorical_covariate_key in obs.columns, f"Error: Categorical covariate {categorical_covariate_key} not found in obs[{idx}]: {obs}"
            # Missing values are left out, so they never become categories such as 'nan', 'None' or '<NA>'
            labels |= set([str(_) for _ in obs[categorical_covariate_key].dropna().unique()])
        labels -= {unlabeled}
        assert len(labels) > 0, f"Error: Categorical covariate {categorical_covariate_key} has no labeled values besides possibly unlabeled: {unlabeled}"

        # Encode from string to integer encodings (for PyTorch)
        sorted_labels = sorted(labels)
        codes, uniques = pd.factorize(sorted_labels)
        obs_encoding_dict[categorical_covariate_key] = {unlabeled: -1} | dict(zip(uniques, codes))
        obs_decoding_dict[categorical_covariate_key] = {-1: unlabeled} | dict(zip(codes, uniques))
        counterfactual_categorical_covariates_list.extend([1 / len(codes)] * len(codes))
    counterfactual_categorical_covariates = np.array(counterfactual_categorical_covariates_list, dtype=np.float32)

    return counterfactual_categorical_covariates, obs_encoding_dict, obs_decoding_dict

def categorical_labels(values: pd.Series, unlabeled: str = 'Unknown') -> list[str]:
    """
    The categorical label of each cell, as a key into the encoding of encode_categorical_covariates.
    The encoding's labels are str() of the values Series.unique() returns, so each cell is looked up by its unique value
    rather than converted with str() itself. A dict keyed by the unique values matches cells by Python equality and hash:
        - A float32 cell iterates as a Python float (str: '0.10000000149011612') but matches its numpy key (str: '0.1').
        - Values pandas treats as equal (e.g. 1 and 1.0) share one label, that of the value unique() returns first.
    Missing values (NaN, None, pd.NA) get the unlabeled value as their label, so they are encoded as unlabeled cells are.
    """
    labels = {value: str(value) for value in values.dropna().unique()}
    # Missing values are not keys (NaN is not equal to itself), so they are found with isna() once for the whole column
    missing = values.isna().to_numpy()
    return [unlabeled if is_missing else labels[value] for value, is_missing in zip(values, missing)]

def encode_sparse_continuous_covariates(obs_list: pd.DataFrame | Iterable[pd.DataFrame], sparse_continuous_covariate_keys):
    """
    Currently supports one-hot encoding covariates into sparse form.
    For a given sparse continuous covariate, we expect two columns: the "category" and the "value":
        E.g., ('Drug', 'Dosage (float)')
        This formats provides one category for which drug was applied (in the first column), and the dosage (in the second column).
    Currently, this implementation does NOT support combinations of multiple categories (multiple drugs and multiple dosages per cell).
    This may be supported in a future version if there is a demand for it (which will require extending the current version).

    """
    if not isinstance(obs_list, list):
        obs_list = [obs_list]

    sparse_continuous_covariates_dict = {}
    for (category_column, value_column) in sparse_continuous_covariate_keys:
        covariate_values = set()
        for obs in obs_list:
            covariate_values.update(obs[category_column])
        covariate_values = sorted(covariate_values)
        sparse_continuous_covariates_dict[category_column] = len(covariate_values), {v:k for k,v in enumerate(covariate_values)}

    return sparse_continuous_covariates_dict

def get_unseen_values(values, known_values):
    """
    Count the values that are not among known_values, e.g. query categories not seen in training.
    Returns {unseen value: number of cells}, with every missing value (NaN or None) counted under one key, NaN.
    Values are compared as dict keys are, so int 1 and float 1.0 are the same value, but str '1' is not.
    Decides nothing: the caller chooses whether unseen values warn or raise.
    """
    values = pd.Series(values)
    unseen = values[~values.isin(known_values)]

    # Unused categories of a categorical column are counted as 0 cells, so only values that cells have are kept
    counts = unseen.value_counts()
    unseen_counts = {value: int(n_cells) for value, n_cells in counts.items() if n_cells > 0}

    # Missing values are counted separately, so NaN and None share one key
    n_missing = int(unseen.isna().sum())
    if n_missing > 0:
        unseen_counts[np.nan] = n_missing
    return unseen_counts

def encode_continuous_covariates(obs_list: pd.DataFrame | Iterable[pd.DataFrame], continuous_covariate_keys, epsilon: float = 1e-5):
    """
    This function z-scores continuous covariates.
    It is not recommended to be used in the PIANO pipeline; rather, the user should normalize continuous covariates ahead of time.
    This avoids the pipeline forcing a normalization that the user does not intend to perform.
    """
    if not isinstance(obs_list, list):
        obs_list = [obs_list]

    # Use law of total expectation and law of total variance
    obs_zscoring_dict = {}
    for continuous_covariate_key in continuous_covariate_keys:
        means, variances, n_cells = [], [], []
        for idx, obs in enumerate(obs_list):
            assert continuous_covariate_key in obs.columns, f"Error: Continuous covariate {continuous_covariate_key} not found in obs[{idx}]: {obs}"
            data = obs[continuous_covariate_key].values.astype(np.float32)
            means.append(data.mean())
            variances.append(data.var())
            n_cells.append(len(obs))

        # X = continuous covariate; Y = dataset; mean_i = E[X|Y], dataset_pmfs = pmf_Y
        means, variances, n_cells = np.array(means), np.array(variances), np.array(n_cells)
        dataset_pmfs = n_cells / np.sum(n_cells)
        # Apply law of total expectation
        # E[E[X|Y]] = Sum(f_Y * E[X|Y])
        mean = np.sum(dataset_pmfs * means)
        # Apply law of total variance
        # var(X) = E[var(X|Y)] + var(E[X|Y])
        # var(X) = Sum(f_Y * (var(X|Y) + (mu_i - mu) ** 2))
        var = np.sum(dataset_pmfs * (variances + (means - mean) ** 2))
        obs_zscoring_dict[continuous_covariate_key] = (mean, var ** 0.5 + epsilon)  # Smoothing variance for stability

    return obs_zscoring_dict
