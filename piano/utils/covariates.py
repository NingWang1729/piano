from typing import Iterable

import numpy as np
import pandas as pd


def encode_categorical_covariates(obs_list: pd.DataFrame | Iterable[pd.DataFrame], obs_columns_to_encode, unlabeled: str = 'Unknown'):
    if not isinstance(obs_list, list):
        obs_list = [obs_list]
    counterfactual = []
    obs_encoding_dict = {}
    obs_decoding_dict = {}
    for col in obs_columns_to_encode:
        # Retrieve set of labels across all datasets
        labels = set()
        for idx, obs in enumerate(obs_list):
            assert col in obs.columns, f"Error: Categorical covariate {col} not found in obs[{idx}]: {obs}"
            labels |= set([str(_) for _ in obs[col].unique()])
        labels -= {unlabeled}
        assert len(labels) > 0, f"Error: Categorical covariate {col} has no labeled values besides possibly unlabeled: {unlabeled}"

        # Encode from string to integer encodings (for PyTorch)
        sorted_labels = sorted(labels)
        codes, uniques = pd.factorize(sorted_labels)
        obs_encoding_dict[col] = {unlabeled: -1} | dict(zip(uniques, codes))
        obs_decoding_dict[col] = {-1: unlabeled} | dict(zip(codes, uniques))
        counterfactual.extend([1 / len(codes)] * len(codes))
    counterfactual_covariates = np.array(counterfactual, dtype=np.float32)

    return counterfactual_covariates, obs_encoding_dict, obs_decoding_dict

def encode_continuous_covariates(obs_list: pd.DataFrame | Iterable[pd.DataFrame], continuous_covariate_keys, epsilon: float = 1e-5):
    if not isinstance(obs_list, list):
        obs_list = [obs_list]

    # Use law of total expectation and law of total variance
    obs_zscoring_dict = {}
    for col in continuous_covariate_keys:
        means, variances, n_cells = [], [], []
        for obs in obs_list:
            data = obs[col].values.astype(np.float32)
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
        obs_zscoring_dict[col] = (mean, var ** 0.5 + epsilon)  # Smoothing variance for stability

    return obs_zscoring_dict
