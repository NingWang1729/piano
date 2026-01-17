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

from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, csc_matrix
from torch.utils.data import Dataset
from pathlib import Path
import statsmodels.api as sm

from torch.cuda import nvtx
from torch.utils.data import Sampler

from piano.utils.triton_sparse import SparseTritonMatrix


# AnnData Dataset Class
class AnnDataset(Dataset):
    def __init__(
        self, adata, memory_mode: Literal['GPU', 'CPU'] = 'GPU',
        categorical_covariate_keys=(), continuous_covariate_keys=(),
        obs_encoding_dict=None, obs_decoding_dict=None, obs_zscoring_dict=None,
    ):
        if obs_encoding_dict is not None or obs_decoding_dict is not None:
            # Must pass in both if creating from Composer class
            assert obs_encoding_dict is not None and obs_decoding_dict is not None
        
        self.length = len(adata.obs.index)

        # Augmented data tensor with adata.X and adata.obs
        aug_data_list = []
        # Convert data to torch.tensor
        if isinstance(adata.X, np.ndarray):
            aug_data_list.append(torch.from_numpy(adata.X).to(torch.float16))
        elif isinstance(adata.X, csr_matrix):
            aug_data_list.append(torch.from_numpy(adata.X.toarray()).to(torch.float16))
        elif isinstance(adata.X, csc_matrix):
            aug_data_list.append(torch.from_numpy(adata.X.toarray()).to(torch.float16).t())
        else:
            # Likely backed data. Not yet supported for training.
            aug_data_list.append(adata.X)

        # Add categorical covariates to augmented matrix list
        self.categorical_covariate_keys = categorical_covariate_keys
        self.continuous_covariate_keys = continuous_covariate_keys
        self.obs = {}
        if obs_encoding_dict is None:
            self.obs_categorical_to_numerical_map = {}
            self.obs_numerical_to_categorical_map = {}

            # Add categorical one-hot encoding matrices
            for col in self.categorical_covariate_keys:
                adata.obs[col] = [str(_) for _ in adata.obs[col]]
                integer_encodings, unique_values = pd.factorize(adata.obs[col])
                aug_data_list.append(F.one_hot(torch.tensor(integer_encodings), len(unique_values)).to(torch.float16))
        else:
            self.obs_categorical_to_numerical_map = obs_encoding_dict
            self.obs_numerical_to_categorical_map = obs_decoding_dict

            # Add categorical one-hot encoding matrices
            for col in self.categorical_covariate_keys:
                max_encoding_modulo = max(self.obs_categorical_to_numerical_map[col].values()) + 1
                integer_encodings = [self.obs_categorical_to_numerical_map[col][_] % max_encoding_modulo for _ in adata.obs[col]]
                aug_data_list.append(F.one_hot(torch.tensor(integer_encodings), max_encoding_modulo).to(torch.float16))

        # Add continouous covariates to augmented matrix list
        if obs_zscoring_dict is None:
            self.obs_zscoring_dict = {}
            for continuous_covariate_key in self.continuous_covariate_keys:
                data = adata.obs[continuous_covariate_key].values.astype(np.float32)
                mean, std = np.mean(data), np.std(data) + np.mean(data) * 1e-5
                self.obs_zscoring_dict[continuous_covariate_key] = (mean, std)
                aug_data_list.append(torch.tensor((data - mean) / std).view(-1, 1))
        else:
            self.obs_zscoring_dict = obs_zscoring_dict
            for continuous_covariate_key in self.continuous_covariate_keys:
                data = adata.obs[continuous_covariate_key].values.astype(np.float32)
                mean, std = self.obs_zscoring_dict[continuous_covariate_key]
                aug_data_list.append(torch.tensor((data - mean) / std).view(-1, 1))

        # Concatenate data
        self.aug_data = torch.hstack(aug_data_list)

        # Move to GPU if available
        if torch.cuda.is_available() and memory_mode == 'GPU':
            self.aug_data = self.aug_data.to(device='cuda', dtype=torch.float32)
        elif memory_mode == 'GPU':
            print("Warning: GPU not available for GPU memory mode.", flush=True)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        nvtx.range_push("AnnDataset.__getitem__")
        aug_data = self.aug_data[index].to(torch.float32)
        nvtx.range_pop()

        return aug_data

    def get_obs_categorical_label_from_numerical_label(self, col, numerical_label):
        return self.obs_numerical_to_categorical_map[col][numerical_label]

class SparseGPUAnnDataset(Dataset):
    def __init__(
        self, adata, memory_mode: Literal['SparseGPU'] = 'SparseGPU',
        categorical_covariate_keys=(), continuous_covariate_keys=(),
        obs_encoding_dict=None, obs_decoding_dict=None, obs_zscoring_dict=None,
    ):
        assert memory_mode == 'SparseGPU', "ERROR: SparseGPUAnnDataset only supports SparseGPU memory mode"
        assert torch.cuda.is_available(), "ERROR: SparseGPUAnnDataset requires having a CUDA GPU available"
        if obs_encoding_dict is not None or obs_decoding_dict is not None:
            # Must pass in both if creating from Composer class
            assert obs_encoding_dict is not None and obs_decoding_dict is not None
        
        self.length = len(adata.obs.index)

        # Augmented data tensor with adata.X and adata.obs
        if not isinstance(adata.X, csr_matrix):
            print(
                "Warning: adata.X is not already sparse. Converting to adata.X to csr_matrix with dtype np.uint16"
                "To use a different dtype, convert adata.X to sparse before creating SparseGPUAnnDataset"
            )
            self.sparse_data = SparseTritonMatrix(csr_matrix(adata.X, dtype=np.uint16))
        else:
            self.sparse_data = SparseTritonMatrix(adata.X)

        # Add categorical covariates to augmented matrix list
        self.categorical_covariate_keys = categorical_covariate_keys
        self.continuous_covariate_keys = continuous_covariate_keys
        self.obs = {}
        aug_data_list = []
        if obs_encoding_dict is None:
            self.obs_categorical_to_numerical_map = {}
            self.obs_numerical_to_categorical_map = {}

            # Add categorical one-hot encoding matrices
            for col in self.categorical_covariate_keys:
                adata.obs[col] = [str(_) for _ in adata.obs[col]]
                integer_encodings, unique_values = pd.factorize(adata.obs[col])
                aug_data_list.append(F.one_hot(torch.tensor(integer_encodings), len(unique_values)).to(torch.float16))
        else:
            self.obs_categorical_to_numerical_map = obs_encoding_dict
            self.obs_numerical_to_categorical_map = obs_decoding_dict

            # Add categorical one-hot encoding matrices
            for col in self.categorical_covariate_keys:
                max_encoding_modulo = max(self.obs_categorical_to_numerical_map[col].values()) + 1
                integer_encodings = [self.obs_categorical_to_numerical_map[col][_] % max_encoding_modulo for _ in adata.obs[col]]
                aug_data_list.append(F.one_hot(torch.tensor(integer_encodings), max_encoding_modulo).to(torch.float16))
        
        # Add continouous covariates to augmented matrix list
        if obs_zscoring_dict is None:
            self.obs_zscoring_dict = {}
            for continuous_covariate_key in self.continuous_covariate_keys:
                data = adata.obs[continuous_covariate_key].values.astype(np.float32)
                mean, std = np.mean(data), np.std(data) + np.mean(data) * 1e-5
                self.obs_zscoring_dict[continuous_covariate_key] = (mean, std)
                aug_data_list.append(torch.tensor((data - mean) / std).view(-1, 1))
        else:
            self.obs_zscoring_dict = obs_zscoring_dict
            for continuous_covariate_key in self.continuous_covariate_keys:
                data = adata.obs[continuous_covariate_key].values.astype(np.float32)
                mean, std = self.obs_zscoring_dict[continuous_covariate_key]
                aug_data_list.append(torch.tensor((data - mean) / std).view(-1, 1))

        # Concatenate data
        self.aug_data = torch.hstack(aug_data_list)
        
        # Move to GPU
        self.aug_data = self.aug_data.to(device='cuda')
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        nvtx.range_push("AnnDataset.__getitem__")
        aug_data = torch.hstack([
            self.sparse_data[index].to(torch.float32), 
            self.aug_data[index].to(torch.float32), 
        ])
        nvtx.range_pop()

        return aug_data
    
    def get_obs_categorical_label_from_numerical_label(self, col, numerical_label):
        return self.obs_numerical_to_categorical_map[col][numerical_label]

class BackedAnnDataset(Dataset):
    """
    Backed ('r') AnnDataset object supporting random-access for true cell-level
    shuffling via __getitem__.

    BackedAnnDataset uses lazy initialization and uses all genes by default
    - To subset to certain genes, call set_var_subset()

    Each worker keeps its own read-only handle to the HDF5 file.
    All tensors are returned on CPU; Composer moves whole batches to CUDA.
    """
    def __init__(
        self,
        adata_path: str | Path, memory_mode: Literal['backed'] = 'backed',
        categorical_covariate_keys=(), continuous_covariate_keys=(),
        obs_encoding_dict=None, obs_decoding_dict=None, obs_zscoring_dict=None, 
    ):
        assert memory_mode == 'backed', "ERROR: BackedAnnDataset only supports backed memory mode"
        self.adata_path = str(adata_path)
        self.categorical_covariate_keys  = list(categorical_covariate_keys)
        self.continuous_covariate_keys = list(continuous_covariate_keys)
        self._adata_backed = sc.read_h5ad(self.adata_path, backed="r")
        if hasattr(self._adata_backed.X, "toarray"):
            self.sparse = True
        else:
            self.sparse = False
        self.n_obs = self._adata_backed.n_obs
        self.var_subset = None

        # Store aug matrix
        aug_data_list = []
        if obs_encoding_dict is None:
            self.obs_categorical_to_numerical_map = {}
            self.obs_numerical_to_categorical_map = {}

            # Add categorical one-hot encoding matrices
            for col in self.categorical_covariate_keys:
                self._adata_backed.obs[col] = [str(_) for _ in self._adata_backed.obs[col]]
                integer_encodings, unique_values = pd.factorize(self._adata_backed.obs[col])
                aug_data_list.append(F.one_hot(torch.tensor(integer_encodings), len(unique_values)).to(torch.float16))
        else:
            self.obs_categorical_to_numerical_map = obs_encoding_dict
            self.obs_numerical_to_categorical_map = obs_decoding_dict

            # Add categorical one-hot encoding matrices
            for col in self.categorical_covariate_keys:
                max_encoding_modulo = max(self.obs_categorical_to_numerical_map[col].values()) + 1
                integer_encodings = [self.obs_categorical_to_numerical_map[col][_] % max_encoding_modulo for _ in self._adata_backed.obs[col]]
                aug_data_list.append(F.one_hot(torch.tensor(integer_encodings), max_encoding_modulo).to(torch.float16))

        # Add continouous covariates to augmented matrix list
        if obs_zscoring_dict is None:
            self.obs_zscoring_dict = {}
            for continuous_covariate_key in self.continuous_covariate_keys:
                data = self._adata_backed.obs[continuous_covariate_key].values.astype(np.float32)
                mean, std = np.mean(data), np.std(data) + np.mean(data) * 1e-5
                self.obs_zscoring_dict[continuous_covariate_key] = (mean, std)
                aug_data_list.append(torch.tensor((data - mean) / std).view(-1, 1))
        else:
            self.obs_zscoring_dict = obs_zscoring_dict
            for continuous_covariate_key in self.continuous_covariate_keys:
                data = self._adata_backed.obs[continuous_covariate_key].values.astype(np.float32)
                mean, std = self.obs_zscoring_dict[continuous_covariate_key]
                aug_data_list.append(torch.tensor((data - mean) / std).view(-1, 1))

        self.aug_data = torch.hstack(aug_data_list)
        print(f"Created aug data for backed dataset with shape: {self.aug_data.shape}")

        # Initialize Dataset __getitem__ to use full genes until .set_var_subset() is called
        if self.sparse:
            self.__getitem__ = self._getitem_sparse_full
        else:
            self.__getitem__ = self._getitem_dense_full

    def set_var_subset(self, var_subset: np.ndarray):
        self.var_subset = var_subset
        self.var_indices = np.arange(len(self._adata_backed.var_names))[np.isin(self._adata_backed.var_names, var_subset)]

        if self.sparse:
            self.__getitem__ = self._getitem_sparse_subset
        else:
            self.__getitem__ = self._getitem_dense_subset

    def __getstate__(self):
        """Pickle-safe: drop live HDF5 handle."""
        state = self.__dict__.copy()
        state["_adata"] = None
        return state

    def __len__(self) -> int:
        return self.n_obs

    def _getitem_sparse_full(self, idx: int) -> torch.Tensor:
        X = self._adata_backed[idx].X
        gene = torch.from_numpy(X.toarray()).float()

        return torch.hstack([gene, self.aug_data[idx]])

    def _getitem_sparse_subset(self, idx: int) -> torch.Tensor:
        X = self._adata_backed[idx].X
        gene = torch.from_numpy(X.toarray()).float()
        gene = gene[:, self.var_indices]

        return torch.hstack([gene, self.aug_data[idx]])

    def _getitem_dense_full(self, idx: int) -> torch.Tensor:
        X = self._adata_backed[idx].X
        gene = torch.from_numpy(np.asarray(X)).float()

        return torch.hstack([gene, self.aug_data[idx]])

    def _getitem_dense_subset(self, idx: int) -> torch.Tensor:
        X = self._adata_backed[idx].X
        gene = torch.from_numpy(np.asarray(X)).float()
        gene = gene[:, self.var_indices]

        return torch.hstack([gene, self.aug_data[idx]])

class GPUBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        # Generate shuffled indices
        indices = torch.randperm(len(self.data_source), device='cuda')

        # Yield batches
        for idx in range(self.__len__()):
            yield indices[idx * self.batch_size:(idx + 1) * self.batch_size]

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

def streaming_hvg_indices(adata, n_top_genes, chunk_size=10_000, span=0.3):
    """
    Seurat-v3–style ("vst") highly-variable-gene (HVG) selection for backed AnnData.
    CPU-based.
    
    Parameters
    ----------
    adata : AnnData (backed)
        `.X` must contain **raw counts** (integer UMI matrix).
    n_top_genes : int
        Number of HVGs to return.
    chunk_size : int, optional (default=10 000)
        Number of cells to load per iteration.
    span : float, optional (default 0.3)
        Fraction of genes used as the LOWESS smoothing window (`frac`
        argument in `statsmodels.nonparametric.lowess`). Set to 0.3 to
        match Scanpy/scVI defaults (their implementation of Seurat v3 "vst").

    Returns
    -------
    np.ndarray
        Integer array of length `n_top_genes` with column indices
        (0-based) of the selected HVGs, sorted from lowest to highest index.
    """
    n_cells, n_genes = adata.n_obs, adata.n_vars
    sum_x  = np.zeros(n_genes, dtype=np.float64)
    sum_x2 = np.zeros(n_genes, dtype=np.float64)

    # compute gene-wise mean, var on log-normalised data
    # -> chunking by cells but practically I don't see why we 
    #    couldn't chunk the features instead
    for start in range(0, n_cells, chunk_size):
        stop = min(start + chunk_size, n_cells)
        X = adata.X[start:stop]

        # get dense array (OK — chunk_size keeps memory bounded)
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=np.float64)

        # library-size normalisation (scale factor 10k, doesn't matter but Seurat default)
        lib = X.sum(1, keepdims=True)
        # avoid divide-by-zero when a cell has zero reads (shouldn't happen ever)
        lib[lib == 0] = 1.0
        X = np.log1p(X / lib * 1e4)

        sum_x  += X.sum(0)
        sum_x2 += (X ** 2).sum(0)

    mu  = sum_x / n_cells
    var = sum_x2 / n_cells - mu**2

    # LOWESS trend fit in log-log space
    # -> Seurat uses LOESS, but scVI subs LOWESS
    good = (var > 0) & (mu > 0)
    log_mu  = np.log10(mu[good])
    log_var = np.log10(var[good])

    # frac=0.3 matches Seurat/Scanpy default; adjust for smoother/rougher fit
    trend = sm.nonparametric.lowess(
        endog=log_var,
        exog=log_mu,
        frac=span,   # 30 % of genes
        it=0,        # same as Scanpy
        return_sorted=False
    )
    # residuals = observed − predicted log(variance)
    resid = np.full(n_genes, -np.inf, dtype=np.float64)
    resid[good] = log_var - trend

    # pick top genes by residual
    top_idx = np.argpartition(resid, -n_top_genes)[-n_top_genes:]
    # optional: sort the output indices (not required by downstream code)
    return np.sort(top_idx)
