import numpy as np
import pandas as pd


def encode_categorical_covariates(obs: pd.DataFrame, obs_columns_to_encode, unlabeled: str = 'Unknown'):
    counterfactual = []
    obs_encoding_dict = {}
    obs_decoding_dict = {}
    for col in obs_columns_to_encode:
        sorted_labels = sorted(set([str(_) for _ in obs[col].unique()]) - {unlabeled})
        codes, uniques = pd.factorize(sorted_labels)
        obs_encoding_dict[col] = {unlabeled: -1} | dict(zip(uniques, codes))
        obs_decoding_dict[col] = {-1: unlabeled} | dict(zip(codes, uniques))
        counterfactual.extend([1 / len(codes)] * len(codes))
    counterfactual_covariates = np.array(counterfactual)

    return counterfactual_covariates, obs_encoding_dict, obs_decoding_dict

def encode_continuous_covariates(obs: pd.DataFrame, continuous_covariate_keys, epsilon: float = 1e-5):
    obs_zscoring_dict = {}
    for col in continuous_covariate_keys:
        data = obs[col].values.astype(np.float32)
        mean = data.mean()
        std = data.std() + mean * epsilon  # Smoothing for stability if low variance
        obs_zscoring_dict[col] = (mean, std)

    return obs_zscoring_dict
