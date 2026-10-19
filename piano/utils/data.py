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
import os
from typing import Iterable, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from anndata.abc import CSCDataset
from scipy.sparse import csr_matrix, csc_matrix, csr_array, csc_array, issparse
from torch.utils.data import Dataset
from pathlib import Path
import statsmodels.api as sm

from torch.cuda import nvtx
from torch.utils.data import Sampler

from piano.utils.covariates import encode_categorical_covariates, encode_sparse_continuous_covariates, encode_continuous_covariates
from piano.utils.triton_sparse import SparseTritonMatrix


class AnnDataset(Dataset):
    def __init__(
        self, adata, memory_mode: Literal['GPU', 'CPU'] = 'GPU',
        categorical_covariate_keys=(), continuous_covariate_keys=(),
        obs_encoding_dict=None, obs_decoding_dict=None,
        stratify_column=None, unlabeled='Unknown',
    ):
        self._initialize_metadata(memory_mode, adata.obs, unlabeled, obs_encoding_dict, obs_decoding_dict)
        self._initialize_stratification(stratify_column)

        # Counts are stored densely as uint16 when they are integers, else as float32 (see to_stored_dense)
        if isinstance(adata.X, np.ndarray):
            counts = to_stored_dense(adata.X)
        elif isinstance(adata.X, (csr_matrix, csc_matrix, csr_array, csc_array)):
            counts = to_stored_csr(adata.X).toarray()  # Converts only the stored values, then fills in the zeros
        else:
            raise TypeError(f"adata.X is a {type(adata.X).__name__}; GPU and CPU memory modes take a numpy array or a scipy sparse matrix")
        self.counts = counts_to_tensor(counts)
        self._read_counts = read_uint16_counts if counts.dtype == np.uint16 else read_float32_counts

        # Covariates are kept separately, as float32
        covariates = self._initialize_covariates([], categorical_covariate_keys, continuous_covariate_keys, obs_encoding_dict, obs_decoding_dict)
        self.aug_data = self._stack_covariates(covariates)
        self._move_to_gpu()
        self.counts = self.counts.to(self.aug_data.device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        nvtx.range_push("AnnDataset.__getitem__")
        genes = self._read_counts(self.counts[index])
        aug_data = torch.hstack([genes, self.aug_data[index]])
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
        obs_encoding_dict, obs_decoding_dict,
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

        # Add continouous covariates to augmented matrix list
        for covariate in self.continuous_covariate_keys:
            aug_data_list.append(self._get_continuous_augmented_matrix(covariate))

        return aug_data_list

    def _stack_covariates(self, covariates):
        """
        Stack encoded covariate columns into one (cells x covariates) float32 tensor.

        Without covariates this is an empty (cells x 0) tensor, as torch.hstack rejects an empty list.
        """
        if not covariates:
            return torch.zeros((self.length, 0))
        return torch.hstack(covariates)
    
    def _initialize_stratification(self, stratify_column):
        self.stratify_column = stratify_column
        if stratify_column is not None:
            labels = self.obs[stratify_column].to_numpy()
            classes, encoded = np.unique(labels, return_inverse=True)
            self.classes = classes
            self.labels = torch.from_numpy(encoded)
            self.class_indices = [torch.where(self.labels == i)[0] for i in range(len(classes))]
            self.class_sizes = [len(idx) for idx in self.class_indices]
        else:
            self.classes = None
            self.labels = None
            self.class_indices = None
            self.class_sizes = None

    def _move_to_gpu(self):
        if self.memory_mode not in ("GPU", "SparseGPU"):
            return
        if torch.cuda.is_available():
            self.aug_data = self.aug_data.to(device='cuda', dtype=torch.float32)
            if self.labels is not None:
                self.labels = self.labels.cuda()
                self.class_indices = [x.cuda() for x in self.class_indices]
        else:
            print(f"Warning: GPU not available for {self.memory_mode} memory mode.", flush=True)

    def _get_categorical_augmented_matrix(self, covariate: str):
        num_categories = max(self.obs_encoding_dict[covariate].values()) + 1
        obs_encodings = np.array([self.obs_encoding_dict[covariate][str(_)] for _ in self.obs[covariate]])
        invalid_mask = (obs_encodings < 0) | (obs_encodings >= num_categories)
        obs_encodings = np.mod(obs_encodings, num_categories)
        aug_matrix = F.one_hot(torch.tensor(obs_encodings), num_categories).to(torch.float32)
        aug_matrix[invalid_mask] = 0

        return aug_matrix

    def _get_continuous_augmented_matrix(self, covariate: str):
        return torch.tensor(self.obs[covariate].values, dtype=torch.float32).view(-1, 1)  # .values is slightly faster

    def get_obs_categorical_label_from_numerical_label(self, col, numerical_label):
        return self.obs_decoding_dict[col][numerical_label]

UINT16_MAX = np.iinfo(np.uint16).max

def counts_to_tensor(counts):
    """
    A torch tensor holding stored counts (a uint16 or float32 array) for indexing in batches.

    torch cannot index uint16 tensors ("index_cuda" not implemented for 'UInt16'), so uint16 counts are kept as the same
    bits in an int16 tensor; read_uint16_counts reads them back.
    """
    if counts.dtype == np.uint16:
        return torch.from_numpy(counts.view(np.int16))
    return torch.from_numpy(counts)

def read_uint16_counts(counts):
    """float32 counts from a batch of uint16 counts stored as int16 bits (see counts_to_tensor)."""
    return counts.view(torch.uint16).to(torch.float32)

def read_float32_counts(counts):
    """A batch of counts stored as float32, as they are."""
    return counts

def _count_non_integers(values):
    """How many counts are not integers (0 for integer dtypes); any decides whether counts can be stored as uint16."""
    if not np.issubdtype(values.dtype, np.floating):
        return 0
    return int(np.count_nonzero(values != np.rint(values)))

def _clip_to_uint16(values):
    """
    Cast counts to uint16, clipping those above 65535 (the largest uint16), which a direct cast would wrap around
    (70000 to 4464). values is left unchanged. Returns the uint16 values and how many counts were clipped.
    """
    n_clipped = np.count_nonzero(values > UINT16_MAX)
    return np.minimum(values, UINT16_MAX).astype(np.uint16), n_clipped

def _warn_about_clipping(n_clipped):
    """Report how many counts _clip_to_uint16 clipped."""
    if n_clipped > 0:
        print(f"Warning: {n_clipped} counts in adata.X are above {UINT16_MAX}, the largest uint16 value; clipping them to {UINT16_MAX}", flush=True)

def _notice_float32_storage(n_non_integer):
    """Tell the user non-integer counts are stored as float32, which takes twice the memory of uint16."""
    print(
        f"Note: {n_non_integer} values in adata.X are not integers, so counts are stored as float32 (twice the memory of uint16). "
        "piano models raw counts: if adata.X was normalized by mistake, pass raw counts instead; if non-integer values are "
        "intended (e.g. TPM from SMART-seq), ignore this note", flush=True,
    )

def _sum_duplicate_entries(X):
    """
    Sum the duplicate entries of a CSR matrix (several stored values for the same cell and gene, whose value scipy
    defines as their sum), so that each count is checked and clipped as a whole. Returns X itself when it is canonical,
    else a summed copy (X may share its arrays with adata.X, and sum_duplicates changes them in place), and how many
    duplicate entries were merged.
    """
    if X.has_canonical_format:
        return X, 0
    summed = X.copy()
    summed.sum_duplicates()
    return summed, X.nnz - summed.nnz

def _notice_duplicate_entries(n_duplicates):
    """Tell the user duplicate entries were summed in a copy of adata.X, and how to avoid the copy."""
    if n_duplicates > 0:
        print(
            f"Note: adata.X has {n_duplicates} duplicate entries (several stored values for the same cell and gene); they are "
            "summed, as scipy does, in a copy of adata.X. To avoid the copy, call adata.X.sum_duplicates() first", flush=True,
        )

def to_stored_csr(X):
    """
    Convert counts (any scipy sparse format, or dense) to the CSR matrix that sparse memory modes store.

    The stored values are checked first, whatever their dtype: if every one is an integer (e.g. raw counts stored as
    float32), counts are stored as uint16 to save memory, clipped to 65535 with a warning (_clip_to_uint16); if any value
    is not an integer, all are stored as float32, unrounded, with a note. Duplicate entries are summed first, with a note
    (_sum_duplicate_entries). X is left unchanged.
    """
    X, n_duplicates = _sum_duplicate_entries(csr_matrix(X))
    _notice_duplicate_entries(n_duplicates)
    n_non_integer = _count_non_integers(X.data)
    if n_non_integer > 0:
        _notice_float32_storage(n_non_integer)
        data = X.data.astype(np.float32)  # A copy, even if X.data is float32 already (X may share it with adata.X)
    else:
        data, n_clipped = _clip_to_uint16(X.data)
        _warn_about_clipping(n_clipped)
    return csr_matrix((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)  # Copies: X may share them with adata.X

def to_stored_dense(X, chunk_size=8192):
    """
    Convert dense counts to the array that GPU and CPU memory modes store, with the same rules as to_stored_csr.

    The values are checked first, whatever their dtype: if every one is an integer (e.g. raw counts stored as float32),
    counts are stored as uint16 to save memory, clipped to 65535 with a warning; if any value is not an integer, all are
    stored as float32, unrounded, with a note. Rows are read in chunks (once to check for non-integers, once to fill the
    stored array), so temporary copies stay small. X is left unchanged.
    """
    chunks = [slice(start, start + chunk_size) for start in range(0, X.shape[0], chunk_size)]
    n_non_integer = sum(_count_non_integers(np.asarray(X[rows])) for rows in chunks)
    if n_non_integer > 0:
        _notice_float32_storage(n_non_integer)
        return np.array(X, dtype=np.float32)  # Always a copy, so adata.X stays unchanged

    counts = np.empty(X.shape, dtype=np.uint16)
    n_clipped = 0
    for rows in chunks:
        counts[rows], n_clipped_rows = _clip_to_uint16(np.asarray(X[rows]))
        n_clipped += n_clipped_rows
    _warn_about_clipping(n_clipped)
    return counts

class SparseGPUAnnDataset(AnnDataset):
    def __init__(
        self, adata, memory_mode: Literal['SparseGPU'] = 'SparseGPU',
        categorical_covariate_keys=(), continuous_covariate_keys=(),
        obs_encoding_dict=None, obs_decoding_dict=None,
        stratify_column=None, unlabeled='Unknown',
    ):
        assert memory_mode == 'SparseGPU', "ERROR: SparseGPUAnnDataset only supports SparseGPU memory mode"
        assert torch.cuda.is_available(), "ERROR: SparseGPUAnnDataset requires having a CUDA GPU available"
        self._initialize_metadata(memory_mode, adata.obs, unlabeled, obs_encoding_dict, obs_decoding_dict)
        self._initialize_stratification(stratify_column)

        # Initialize augmented data tensor with adata.obs
        # Counts are stored as CSR, uint16 when they are integers, else float32 (see to_stored_csr)
        self.sparse_data = SparseTritonMatrix(to_stored_csr(adata.X))
        covariates = self._initialize_covariates([], categorical_covariate_keys, continuous_covariate_keys, obs_encoding_dict, obs_decoding_dict)
        self.aug_data = self._stack_covariates(covariates)
        self._move_to_gpu()

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

class SparseCPUAnnDataset(AnnDataset):
    def __init__(
        self, adata, memory_mode: Literal['SparseCPU'] = 'SparseCPU',
        categorical_covariate_keys=(), continuous_covariate_keys=(),
        sparse_continuous_covariate_keys=(),
        obs_encoding_dict=None, obs_decoding_dict=None,
        sparse_continuous_covariates_dict=None,
        stratify_column=None, unlabeled='Unknown',
    ):
        """
        Only SparseCPUAnnDataset supports sparse_continuous_covariate_keys!
        If necessary for other memory modes, then the data is too big, and will need SparseCPUAnnDataset 
        """
        assert memory_mode == 'SparseCPU', "ERROR: SparseCPUAnnDataset only supports SparseCPU memory mode"
        self._initialize_metadata(memory_mode, adata.obs, unlabeled, obs_encoding_dict, obs_decoding_dict)
        self._initialize_stratification(stratify_column)

        # Initialize augmented data tensor with adata.obs
        # Counts are stored as CSR, uint16 when they are integers, else float32 (see to_stored_csr)
        self.sparse_data = to_stored_csr(adata.X)

        covariates = self._initialize_covariates([], categorical_covariate_keys, continuous_covariate_keys, sparse_continuous_covariate_keys, obs_encoding_dict, obs_decoding_dict, sparse_continuous_covariates_dict)
        self.aug_data = self._stack_covariates(covariates)

    def __len__(self):
        return self.length

    def _initialize_covariates(self, aug_data_list, categorical_covariate_keys, continuous_covariate_keys, sparse_continuous_covariate_keys, obs_encoding_dict, obs_decoding_dict, sparse_continuous_covariates_dict):
        aug_data_list = super()._initialize_covariates(aug_data_list, categorical_covariate_keys, continuous_covariate_keys, obs_encoding_dict, obs_decoding_dict)

        self.sparse_continuous_covariate_keys = sparse_continuous_covariate_keys
        if sparse_continuous_covariates_dict is not None:
            self.sparse_continuous_covariates_dict = sparse_continuous_covariates_dict
        else:
            self.sparse_continuous_covariates_dict = encode_sparse_continuous_covariates(self.obs, self.sparse_continuous_covariate_keys)

        return aug_data_list

    def _extract_sparse_continuous_covariates(self, sparse_continuous_covariates, index):
        n_cells = len(index)
        if len(sparse_continuous_covariates) <= 0:
            return torch.zeros((n_cells, 0), dtype=torch.float32)

        dense_covariate_columns = []
        for (category_column, value_column) in sparse_continuous_covariates:
            # Extract columns for category and value (e.g., drug name and drug dosage)
            n_categories, encoding_dict = self.sparse_continuous_covariates_dict[category_column]
            categories = self.obs.iloc[index][category_column].values
            values = self.obs.iloc[index][value_column].values

            # Create one-hot matrix
            # col_indices = np.array(
            #     [encoding_dict.get(cat, -1) for cat in categories],
            #     dtype=np.int32
            # )
            # col_indices = pd.Series(categories).map(encoding_dict).fillna(-1).to_numpy(dtype=np.int32)
            col_indices = np.array(
                [encoding_dict.get(cat, -1) for cat in categories],
                dtype=np.int32
            )
            valid_mask = col_indices >= 0
            row_indices = np.arange(n_cells)[valid_mask]
            col_indices = col_indices[valid_mask]
            vals = values[valid_mask]
            dense_covariates = np.zeros((n_cells, n_categories), dtype=np.float32)
            dense_covariates[row_indices, col_indices] = vals

            dense_covariate_columns.append(dense_covariates)

        return torch.tensor(
            np.hstack(dense_covariate_columns),
            dtype=torch.float32
        )

    def __getitem__(self, index):
        nvtx.range_push("AnnDataset.__getitem__")
        np_idx = index.cpu().numpy() if isinstance(index, torch.Tensor) else index
        aug_data = torch.hstack([
            torch.tensor(self.sparse_data[np_idx].toarray(), dtype=torch.float32),
            self.aug_data[index].to(torch.float32),
            self._extract_sparse_continuous_covariates(self.sparse_continuous_covariate_keys, np_idx)
        ])
        nvtx.range_pop()

        return aug_data

    def get_obs_categorical_label_from_numerical_label(self, col, numerical_label):
        return self.obs_decoding_dict[col][numerical_label]

class BackedAnnDataset(AnnDataset):
    """
    AnnDataset whose counts stay on disk, in a .h5ad file opened in anndata's backed mode.

    Accepts a .h5ad path or an AnnData read with read_h5ad(path, backed='r'). Only obs, var and the
    encoded covariates are held in memory; each mini-batch reads only its own cells' counts from disk.
    All genes are read until set_var_subset() selects the genes to use.

    An open file cannot be pickled, so the dataset keeps the file's path and opens it on first use in
    each process (e.g. in each DataLoader worker).

    Reads use anndata's backed row indexing (about 145 batches of 128 random cells per second from a CSR
    .h5ad of 17k cells x 4k genes). A reader that reads each stored chunk once per batch could be
    faster; see the TODO list.
    """
    def __init__(
        self, adata, memory_mode: Literal['backed'] = 'backed',
        categorical_covariate_keys=(), continuous_covariate_keys=(),
        obs_encoding_dict=None, obs_decoding_dict=None,
        stratify_column=None, unlabeled='Unknown',
    ):
        assert memory_mode == 'backed', "ERROR: BackedAnnDataset only supports backed memory mode"
        self.filename = backed_h5ad_path(adata)
        self._adata = None  # Opened on first use in each process
        if isinstance(self.adata.X, CSCDataset):
            raise ValueError(
                f"{self.filename} stores X as CSC, so reading any cell would read the whole matrix. "
                "Rewrite it as CSR (adata.X = adata.X.tocsr(); adata.write_h5ad(...)), or use an in-memory memory mode"
            )
        self._initialize_metadata(memory_mode, self.adata.obs, unlabeled, obs_encoding_dict, obs_decoding_dict)
        self._initialize_stratification(stratify_column)
        self.var_indices = None  # All genes, in file order

        # Only the covariates are held in memory, as the rest of each training row
        covariates = self._initialize_covariates([], categorical_covariate_keys, continuous_covariate_keys, obs_encoding_dict, obs_decoding_dict)
        self.aug_data = self._stack_covariates(covariates)

    @property
    def adata(self):
        """The backed AnnData: obs and var in memory, counts on disk."""
        if self._adata is None:
            self._adata = ad.read_h5ad(self.filename, backed='r')
        return self._adata

    def set_var_subset(self, var_subset):
        """Read only these genes, in this order (the order Composer.var_names gives the in-memory modes too)."""
        var_indices = self.adata.var_names.get_indexer(var_subset)
        if (var_indices < 0).any():
            missing = np.asarray(var_subset)[var_indices < 0]
            raise ValueError(f"{len(missing)} genes are missing from {self.filename}, e.g. {list(missing[:5])}")
        self.var_indices = var_indices

    def __getstate__(self):
        """Pickle without the open file, which each process reopens on first use."""
        state = self.__dict__.copy()
        state['_adata'] = None
        return state

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        nvtx.range_push("BackedAnnDataset.__getitem__")
        rows = torch.atleast_1d(torch.as_tensor(index).cpu()).numpy()  # A batch of cell indices; a single index gives a batch of one

        # Read each distinct cell once, in increasing order (as HDF5 needs), then put the cells back in the requested order
        unique_rows, inverse = np.unique(rows, return_inverse=True)
        X = self.adata.X[unique_rows]
        if self.var_indices is not None:
            X = X[:, self.var_indices]
        X = X[inverse]

        genes = torch.from_numpy(X.toarray() if issparse(X) else np.asarray(X)).to(torch.float32)
        aug_data = torch.hstack([genes, self.aug_data[torch.from_numpy(rows)]])
        nvtx.range_pop()

        return aug_data


def backed_h5ad_path(adata) -> str:
    """
    Path of the .h5ad file that backed mode reads counts from: a .h5ad path, or the file of an AnnData
    read with read_h5ad(path, backed='r'). AnnDatas in memory are refused, since backed mode keeps counts on disk.
    """
    if isinstance(adata, (str, os.PathLike)):
        if not str(adata).endswith('.h5ad'):
            raise ValueError(f"Backed mode reads .h5ad files; got {adata}")
        return str(adata)
    if isinstance(adata, ad.AnnData) and adata.isbacked:
        if adata.is_view:
            raise ValueError("Backed AnnData views are not supported; pass the full backed AnnData or its .h5ad path")
        return str(adata.filename)
    raise ValueError(
        "Backed mode keeps counts on disk: pass a .h5ad path or an AnnData read with read_h5ad(path, backed='r'). "
        "For an AnnData in memory, use memory_mode 'GPU', 'SparseGPU', 'CPU' or 'SparseCPU'"
    )

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
        self._initialize_stratification()

    def _initialize_stratification(self):
        # Without a stratify_column, the datasets have no class labels to combine
        self.classes = None
        self.class_indices = None
        self.class_sizes = None
        if any(dataset.class_indices is None for dataset in self.datasets):
            return

        # Each dataset numbers its own classes, so the same number can mean different classes across datasets.
        # Match classes by name, shifting each dataset's local indices by where that dataset starts in the concatenation.
        dataset_starts = [0] + self.cumulative_sizes[:-1]
        indices_by_class = {}
        for dataset, start in zip(self.datasets, dataset_starts):
            for class_name, local_indices in zip(dataset.classes, dataset.class_indices):
                indices_by_class.setdefault(class_name, []).append(local_indices.to(self.device) + start)
        self.classes = np.array(sorted(indices_by_class))
        self.class_indices = [torch.cat(indices_by_class[class_name]) for class_name in self.classes]
        self.class_sizes = [len(idx) for idx in self.class_indices]

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

def count_cells_per_epoch(n_cells, class_sizes=None, samples_per_class=None):
    """
    Number of cells an epoch draws: all n_cells, or with stratified sampling (samples_per_class set) at most
    samples_per_class cells of each class, whose sizes are class_sizes.
    """
    if samples_per_class is None:
        return n_cells
    return sum(min(samples_per_class, class_size) for class_size in class_sizes)

def count_batches(n_cells_per_epoch, batch_size, drop_last):
    """
    Number of batches in an epoch of n_cells_per_epoch cells; drop_last drops the last batch if it is incomplete.
    """
    if drop_last:
        return n_cells_per_epoch // batch_size
    return (n_cells_per_epoch + batch_size - 1) // batch_size  # Rounds up

class TensorBatchSampler(Sampler):
    def __init__(
        self,
        data_source,
        batch_size,
        device: Literal["cpu", "cuda"] = "cpu",
        shuffle: bool = True,
        drop_last: bool = False,
        samples_per_class: int | None = None,
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.samples_per_class = samples_per_class

        # Stratified sampling needs class labels, which the dataset only builds when given a stratify_column
        if samples_per_class is not None and getattr(data_source, 'class_indices', None) is None:
            raise ValueError("samples_per_class requires a dataset built with a stratify_column; set stratify_column or leave samples_per_class as None")

        # Toggle stratified sampling
        if samples_per_class is None:
            self._get_indices = self._get_all_indices
            class_sizes = None
        else:
            self._get_indices = self._get_stratified_indices
            class_sizes = [len(idx) for idx in data_source.class_indices]
        self.total_samples = count_cells_per_epoch(len(data_source), class_sizes, samples_per_class)  # Total number of cells per epoch, not number of batches
        self.num_batches = count_batches(self.total_samples, batch_size, drop_last)
        self.n_cells_per_epoch = self.num_batches * batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # split() of no indices still gives one empty batch, so an epoch without batches yields nothing
        if self.num_batches == 0:
            return
        indices = self._get_indices()
        if self.drop_last:
            indices = indices[:self.n_cells_per_epoch]
        yield from indices.split(self.batch_size)

    def _get_all_indices(self):
        if self.shuffle:
            return torch.randperm(len(self.data_source), device=self.device)
        return torch.arange(len(self.data_source), device=self.device)

    def _get_stratified_indices(self):
        sampled = []
        for class_indices in self.data_source.class_indices:
            idx = class_indices
            if self.shuffle:
                idx = idx[torch.randperm(len(idx), device=self.device)]
            idx = idx[:self.samples_per_class]
            sampled.append(idx)
        indices = torch.cat(sampled)
        if self.shuffle:
            indices = indices[torch.randperm(len(indices), device=self.device)]
        return indices

def streaming_hvg_indices(adata, n_top_genes, chunk_size=10_000, span=0.3):
    """
    Highly-variable-gene (HVG) selection for backed AnnData from the mean and variance of
    log-normalized counts, with a LOWESS trend and no batch key. CPU-based.

    This is not Seurat v3 "vst", which standardizes raw counts with clipping, and it selects
    different genes than piano.highly_variable_genes. Composer selects backed genes with
    highly_variable_genes, which reads backed counts in chunks.
    
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
