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
            labels |= set([str(_) for _ in obs[categorical_covariate_key].unique()])
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
    for (category, value) in sparse_continuous_covariate_keys:
        covariate_values = set()
        for obs in obs_list:
            covariate_values.update(obs[category])
        covariate_values = sorted(covariate_values)
        sparse_continuous_covariates_dict[category] = len(covariate_values), {v:k for k,v in enumerate(covariate_values)}

    return sparse_continuous_covariates_dict

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
