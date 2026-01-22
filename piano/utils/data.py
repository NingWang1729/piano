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

import bisect
from typing import Iterable, Literal

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

from piano.utils.covariates import encode_categorical_covariates, encode_continuous_covariates
from piano.utils.triton_sparse import SparseTritonMatrix


class AnnDataset(Dataset):
    def __init__(
        self, adata, memory_mode: Literal['GPU', 'CPU'] = 'GPU',
        categorical_covariate_keys=(), continuous_covariate_keys=(),
        obs_encoding_dict=None, obs_decoding_dict=None, obs_zscoring_dict=None,
        unlabeled='Unknown',
    ):
        self._initialize_metadata(memory_mode, adata.obs, unlabeled, obs_encoding_dict, obs_decoding_dict)

        # Initialize augmented data tensor with adata.X and adata.obs
        aug_data_list = []
        if isinstance(adata.X, np.ndarray):
            aug_data_list.append(torch.from_numpy(adata.X).to(torch.float32))
        elif isinstance(adata.X, csr_matrix):
            aug_data_list.append(torch.from_numpy(adata.X.toarray()).to(torch.float32))
        elif isinstance(adata.X, csc_matrix):
            aug_data_list.append(torch.from_numpy(adata.X.toarray()).to(torch.float32).t())
        else:
            # Likely backed data. Not yet supported for training.
            aug_data_list.append(adata.X)
        self._initialize_covariates(aug_data_list, categorical_covariate_keys, continuous_covariate_keys, obs_encoding_dict, obs_decoding_dict, obs_zscoring_dict)
        self.aug_data = torch.hstack(aug_data_list)

        # Move to GPU
        if memory_mode != 'GPU':
            return
        if torch.cuda.is_available():
            self.aug_data = self.aug_data.to(device='cuda', dtype=torch.float32)
        else:
            print("Warning: GPU not available for GPU memory mode.", flush=True)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        nvtx.range_push("AnnDataset.__getitem__")
        aug_data = self.aug_data[index].to(torch.float32)
        nvtx.range_pop()

        return aug_data

    def _initialize_metadata(self, memory_mode, obs, unlabeled, obs_encoding_dict, obs_decoding_dict):
        self.memory_mode = memory_mode

        # Must pass in both encoding and decoding dicts if creating from Composer class
        if obs_encoding_dict is not None or obs_decoding_dict is not None:
            assert obs_encoding_dict is not None and obs_decoding_dict is not None

        self.length = len(obs.index)
        self.obs = obs
        self.unlabeled = unlabeled

    def _initialize_covariates(
        self, aug_data_list: list,
        categorical_covariate_keys, continuous_covariate_keys,
        obs_encoding_dict, obs_decoding_dict, obs_zscoring_dict,
    ):
        # Add valid one-hot encodings and invalid encodings as zeros to augmented matrix list
        self.categorical_covariate_keys = categorical_covariate_keys
        self.continuous_covariate_keys = continuous_covariate_keys
        if obs_encoding_dict is None or obs_decoding_dict is None:
            _, self.obs_encoding_dict, self.obs_decoding_dict = encode_categorical_covariates(self.obs, self.categorical_covariate_keys, self.unlabeled)
        else:
            self.obs_encoding_dict, self.obs_decoding_dict = obs_encoding_dict, obs_decoding_dict
        for covariate in self.categorical_covariate_keys:
            aug_data_list.append(self._get_categorical_augmented_matrix(covariate))

        # Add Z-scored continouous covariates to augmented matrix list
        if obs_zscoring_dict is None:
            self.obs_zscoring_dict = encode_continuous_covariates(self.obs, self.continuous_covariate_keys)
        else:
            self.obs_zscoring_dict = obs_zscoring_dict
        for covariate in self.continuous_covariate_keys:
            aug_data_list.append(self._get_continuous_augmented_matrix(covariate))

        return aug_data_list

    def _get_categorical_augmented_matrix(self, covariate: str):
        num_categories = max(self.obs_encoding_dict[covariate].values()) + 1
        obs_encodings = np.array([self.obs_encoding_dict[covariate][str(_)] for _ in self.obs[covariate]])
        invalid_mask = (obs_encodings < 0) | (obs_encodings >= num_categories)
        obs_encodings = np.mod(obs_encodings, num_categories)
        aug_matrix = F.one_hot(torch.tensor(obs_encodings), num_categories).to(torch.float32)
        aug_matrix[invalid_mask] = 0

        return aug_matrix

    def _get_continuous_augmented_matrix(self, covariate: str):
        data = self.obs[covariate].values.astype(np.float32)
        mean, std = self.obs_zscoring_dict[covariate]

        return torch.tensor((data - mean) / std).view(-1, 1)

    def get_obs_categorical_label_from_numerical_label(self, col, numerical_label):
        return self.obs_decoding_dict[col][numerical_label]

class SparseGPUAnnDataset(AnnDataset):
    def __init__(
        self, adata, memory_mode: Literal['SparseGPU'] = 'SparseGPU',
        categorical_covariate_keys=(), continuous_covariate_keys=(),
        obs_encoding_dict=None, obs_decoding_dict=None, obs_zscoring_dict=None,
        unlabeled='Unknown',
    ):
        assert memory_mode == 'SparseGPU', "ERROR: SparseGPUAnnDataset only supports SparseGPU memory mode"
        assert torch.cuda.is_available(), "ERROR: SparseGPUAnnDataset requires having a CUDA GPU available"
        self._initialize_metadata(memory_mode, adata.obs, unlabeled, obs_encoding_dict, obs_decoding_dict)

        # Initialize augmented data tensor with adata.obs
        if not isinstance(adata.X, csr_matrix):
            print(
                "Warning: adata.X is not already sparse. Converting to adata.X to csr_matrix with dtype np.uint16"
                "To use a different dtype, convert adata.X to sparse before creating SparseGPUAnnDataset"
            )
            self.sparse_data = SparseTritonMatrix(csr_matrix(adata.X, dtype=np.uint16))
        else:
            self.sparse_data = SparseTritonMatrix(adata.X)
        self.aug_data = torch.hstack(
            self._initialize_covariates([], categorical_covariate_keys, continuous_covariate_keys, obs_encoding_dict, obs_decoding_dict, obs_zscoring_dict)
        )

        # Move to GPU
        if torch.cuda.is_available():
            self.aug_data = self.aug_data.to(device='cuda', dtype=torch.float32)
        else:
            print("Warning: GPU not available for GPU memory mode.", flush=True)

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
        return self.obs_decoding_dict[col][numerical_label]

class BackedAnnDataset(AnnDataset):
    """
    Backed ('r') AnnDataset object supporting random-access for true cell-level
    shuffling via __getitem__.

    BackedAnnDataset uses lazy initialization and uses all genes by default
    - To subset to certain genes, call set_var_subset()

    Each worker keeps its own read-only handle to the HDF5 file.
    All tensors are returned on CPU; Composer moves whole batches to CUDA.
    """
    def __init__(
        self, adata, memory_mode: Literal['backed'] = 'backed',
        categorical_covariate_keys=(), continuous_covariate_keys=(),
        obs_encoding_dict=None, obs_decoding_dict=None, obs_zscoring_dict=None,
        unlabeled='Unknown',
    ):
        assert memory_mode == 'backed', "ERROR: BackedAnnDataset only supports backed memory mode"
        self._initialize_metadata(memory_mode, adata.obs, unlabeled, obs_encoding_dict, obs_decoding_dict)

        # Initialize augmented data tensor with adata.obs
        self.adata = adata
        if hasattr(self.adata.X, "toarray"):
            self.sparse = True
        else:
            self.sparse = False
        self.n_obs = self.adata.n_obs
        self.var_subset = None
        self.aug_data = torch.hstack(
            self._initialize_covariates([], categorical_covariate_keys, continuous_covariate_keys, obs_encoding_dict, obs_decoding_dict, obs_zscoring_dict)
        )
        print(f"Created aug data for backed dataset with shape: {self.aug_data.shape}")

        # Initialize Dataset __getitem__ to use full genes until .set_var_subset() is called
        if self.sparse:
            self.__getitem__ = self._getitem_sparse_full
        else:
            self.__getitem__ = self._getitem_dense_full

    def set_var_subset(self, var_subset: np.ndarray):
        self.var_subset = var_subset
        self.var_indices = np.arange(len(self.adata.var_names))[np.isin(self.adata.var_names, var_subset)]

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
        X = self.adata[idx].X
        gene = torch.from_numpy(X.toarray()).float()

        return torch.hstack([gene, self.aug_data[idx]])

    def _getitem_sparse_subset(self, idx: int) -> torch.Tensor:
        X = self.adata[idx].X
        gene = torch.from_numpy(X.toarray()).float()
        gene = gene[:, self.var_indices]

        return torch.hstack([gene, self.aug_data[idx]])

    def _getitem_dense_full(self, idx: int) -> torch.Tensor:
        X = self.adata[idx].X
        gene = torch.from_numpy(np.asarray(X)).float()

        return torch.hstack([gene, self.aug_data[idx]])

    def _getitem_dense_subset(self, idx: int) -> torch.Tensor:
        X = self.adata[idx].X
        gene = torch.from_numpy(np.asarray(X)).float()
        gene = gene[:, self.var_indices]

        return torch.hstack([gene, self.aug_data[idx]])

class ConcatAnnDataset():
    """Dataset as a concatenation of multiple datasets.
    This class is useful to assemble different existing datasets.

    This class is a modification of the PyTorch ConcatDataset:
    - Uses AnnDataset instead of non-IterableDataset (PyTorch)
    - idx checks are removed to support splicing for custom batching

    Args:
        anndatasets (sequence): List of anndatasets to be concatenated
    """
    datasets: list[AnnDataset]
    cumulative_sizes: list[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[AnnDataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        self.cumulative_sizes = self.cumsum(self.datasets)

        memory_modes = set()
        for dataset in self.datasets:
            memory_modes.add(dataset.memory_mode)
        memory_modes = list(memory_modes)
        assert len(memory_modes) == 1, f"Error: ConcatAnnDataset does not support multiple memory modes at once: {memory_modes}"
        self.memory_mode = memory_modes[0]
        match self.memory_mode:
            case 'GPU' | 'SparseGPU':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            case _:
                self.device = 'cpu'
        self.cumulative_sizes_tensor = torch.tensor(self.cumulative_sizes, device=self.device, dtype=torch.long)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # -------- Scalar fast path --------
        if isinstance(idx, int):
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            offset = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
            return self.datasets[dataset_idx][idx - offset]
        else:
            idx = torch.tensor(idx, device=self.device)

        # -------- Map global -> dataset --------
        dataset_indices = torch.searchsorted(self.cumulative_sizes_tensor, idx, right=True)
        offsets = torch.where(dataset_indices > 0, self.cumulative_sizes_tensor[dataset_indices - 1], torch.zeros_like(dataset_indices))
        local_indices = idx - offsets

        # -------- Grouping + fetch --------
        dataset_batches = []
        mini_batch_positions_list = []
        for dataset_index in torch.unique(dataset_indices).tolist():
            mini_batch_positions = torch.nonzero(dataset_indices == dataset_index, as_tuple=True)[0]
            dataset_batch = self.datasets[dataset_index][local_indices[mini_batch_positions]]
            dataset_batches.append(dataset_batch)
            mini_batch_positions_list.append(mini_batch_positions)

        # -------- Concatenate --------
        return torch.cat(dataset_batches, dim=0)[torch.argsort(torch.cat(mini_batch_positions_list, dim=0))]

class GPUBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle: bool = True, drop_last: bool = False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        if drop_last:
            self._len = self._len_drop_last
        else:
            self._len = self._len_no_drop_last

        if shuffle:
            self._iter = self._iter_shuffle
        else:
            self._iter = self._iter_no_shuffle

    def __len__(self):
        return self._len()

    def __iter__(self):
        return self._iter()

    def _len_drop_last(self):
        # Get number of batches to iterate over
        return len(self.data_source) // self.batch_size

    def _len_no_drop_last(self):
        # Get number of batches to iterate over
        # If evenly divisible, adding self.batch_size - 1 does not falsely increase number of batches
        # If not evenly divisible, adding self.batch_size - 1 increases integer division result by 1
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def _iter_shuffle(self):
        # Generate shuffled indices
        indices = torch.randperm(len(self.data_source), device='cuda')
        for idx in range(self.__len__()):
            yield indices[idx * self.batch_size:(idx + 1) * self.batch_size]

    def _iter_no_shuffle(self):
        # Generate sequential indices
        indices = torch.arange(len(self.data_source), device='cuda')
        for idx in range(self.__len__()):
            yield indices[idx * self.batch_size:(idx + 1) * self.batch_size]

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
