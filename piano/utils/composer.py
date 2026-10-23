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

import copy
import inspect
import io
import numbers
import os
import pickle
import random
import warnings
from collections.abc import Iterable
from functools import partial
from typing import Iterable, Literal, Union, List

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from torch.cuda import nvtx
from torch.utils.data import DataLoader
from tqdm import tqdm

from piano.models.base_models import Etude, EtudeMuTheta
from piano.utils.covariates import encode_categorical_covariates, encode_sparse_continuous_covariates, get_unseen_values
from piano.utils.data import AnnDataset, SparseGPUAnnDataset, SparseCPUAnnDataset, BackedAnnDataset, ConcatAnnDataset, TensorBatchSampler, count_batches, count_cells_per_epoch
from piano.utils.preprocessing import DEFAULT_N_TOP_GENES, highly_variable_genes, read_gene_set_genes, _uses_hvg_selection, _select_hvgs_within_gene_set, _warn_n_top_genes_above_gene_set

# Memory mode used instead of each GPU memory mode when no CUDA GPU is available: the CPU mode with the same data layout
CPU_FALLBACK_MEMORY_MODES = {'GPU': 'CPU', 'SparseGPU': 'SparseCPU'}
# Warnings for that fallback, filled in with the memory mode asked for and the one used instead: the default one, for a
# memory_mode argument, and the one of Composer.load, which has no memory_mode argument
MEMORY_MODE_FALLBACK_WARNING = (
    "memory_mode='{memory_mode}' needs a CUDA GPU, but none is available: using memory_mode='{fallback_memory_mode}' "
    "on the CPU instead. Pass memory_mode='{fallback_memory_mode}' to use the CPU without this warning."
)
LOAD_MEMORY_MODE_FALLBACK_WARNING = (
    "This Composer was saved with memory_mode='{memory_mode}', which needs a CUDA GPU, but none is available: using "
    "memory_mode='{fallback_memory_mode}' on the CPU instead."
)


class _DeviceMappingUnpickler(pickle.Unpickler):
    """
    Unpickler that loads the pickled torch tensors to a given device. A pickled tensor is rebuilt by
    torch.storage._load_from_bytes, which calls torch.load without map_location, so a model pickled on a GPU fails to
    unpickle on a machine without one ("Attempting to deserialize object on a CUDA device"). This unpickler swaps that
    function for one that passes map_location=device.
    """
    def __init__(self, file, device):
        super().__init__(file)
        self.device = device

    def find_class(self, module, name):
        if (module, name) == ('torch.storage', '_load_from_bytes'):
            return lambda b: torch.load(io.BytesIO(b), map_location=self.device, weights_only=False)
        return super().find_class(module, name)


class Composer():
    # ======================
    # Construction & I/O
    # ======================
    def __init__(
        self,

        # Training data: AnnData objects (read with read_h5ad(path, backed='r') for memory_mode='backed'). adata.X holds raw
        # counts, which must be finite and non-negative: in-memory counts are checked here, backed counts are not (see _check_adata)
        adata: Union[ad.AnnData | List[ad.AnnData]],

        # Composer arguments
        # Without a CUDA GPU, 'GPU' and 'SparseGPU' fall back to 'CPU' and 'SparseCPU' with a warning
        memory_mode: Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'] = 'GPU',
        compile_model: bool = True,
        # Each column must hold one kind of value (number, str, bool, or one other type), else a ValueError at creation;
        # missing values and the unlabeled value are not a kind, so e.g. [1, 2, 'Unknown'] is accepted; int mixed with
        # float warns, since equal values such as 1 and 1.0 share one category
        categorical_covariate_keys=None,
        continuous_covariate_keys=None,
        sparse_continuous_covariate_keys=None,
        unlabeled: str = 'Unknown',

        # Gene selection
        flavor: str = 'seurat_v3',  # Only Seurat V3 is supported (for multiple AnnDatas)
        # Number of highly variable genes (Seurat v3): None or an integer, else a ValueError at creation.
        # Without a gene set: None means 4096 HVGs and <= 0 means all genes. With a gene set: None or <= 0 means the
        # whole set, and a smaller number that many HVGs ranked within the set (see _select_genes); a number above the
        # genes of the set found warns and uses them all
        n_top_genes: int | None = None,
        hvg_batch_key: str = None,
        geneset_path: str = None,  # File with one gene per line (no header); its genes are used in the AnnData's gene order; read at creation (warns about genes not in the data, raises if none are)
        # The Composer keeps its own AnnData per training AnnData, with only X, obs and var (layers, obsm, obsp, varm and
        # uns are dropped). An in-memory AnnData (not a view) that already has exactly the selected genes in the same order
        # (the whole gene set, or all genes with n_top_genes <= 0 and no gene set) shares its counts (adata.X) with the
        # Composer, to save memory, and a warning says so at creation: do not change adata.X in place while using the
        # Composer. copy_adata=True copies the counts too, so nothing is shared. Other AnnDatas (other genes, HVGs, also
        # HVGs within a gene set, views) are always copied.
        copy_adata: bool = False,

        # Model kwargs
        ## Architecture
        n_hidden: int = 256,
        n_layers: int = 3,
        latent_size: int = 32,
        ## Training mode
        adversarial: bool = True,
        ## Hyperparameters
        dropout_rate: float = 0.1,
        batchnorm_eps: float = 1e-5,       # Torch default is 1e-5
        batchnorm_momentum: float = 1e-1,  # Torch default is 1e-1
        epsilon: float = 1e-5,             # Torch default is 1e-5
        ## Distribution
        distribution: Literal['nb', 'zinb'] = 'nb',
        parameterization: Literal['ksi-psi', 'mu-theta'] = 'ksi-psi',

        # Training
        stratify_column = None,
        samples_per_class: int = None,
        max_epochs: int = 200,
        ## Beta annealing
        batch_size: int = 128,
        min_kld_weight: float = 0.00,
        max_kld_weight: float = 0.25,
        min_adv_weight: float = 1.00,
        max_adv_weight: float = 1.00,
        n_annealing_epochs: int = None,
        ## Hyperparameters
        lr: float = 2e-4,
        weight_decay: float = 0.00,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        ## Early stopping
        early_stopping: bool = True,
        min_delta: float = 1.00,
        patience: int = 5,
        ## Checkpoints
        save_initial_weights: bool = False,
        checkpoint_every_n_epochs = None,

        # Reproducibility
        deterministic: bool = True,
        random_seed: int | None = 0,

        # Output
        run_name: str = 'piano_integration',
        outdir: str = './results/',
    ):
        self._init_params = self._inspect_init_params(locals())
        self._init_pipeline_flags()
        self._init_hardware(self._init_params)
        self._init_data_config(self._init_params)
        self._init_gene_selection(self._init_params)
        self._warn_shared_counts()
        self._init_model_config(self._init_params)
        self._disable_adversarial_without_categorical_covariates()
        self._init_training_config(self._init_params)
        self._init_output_config(self._init_params)
        self._init_random_seed(self._init_params)
        self.set_determinism(deterministic=deterministic, random_seed=random_seed)

    def _inspect_init_params(self, locals_):
        # Uses reflection to capture the input parameters to the constructor
        sig = inspect.signature(self.__init__)

        return {
            name: locals_.get(name, p.default)
            for name, p in sig.parameters.items()
            if name != "self"
        }

    def _init_pipeline_flags(self):
        self.initialized_features = False
        self.prepared_data = False
        self.prepared_model = False
        self.trained_model = False

    def _init_hardware(self, params):
        self.memory_mode = self._canonical_memory_mode(params['memory_mode'])
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.compile_model = params['compile_model']
        else:
            self.device = 'cpu'
            if self.memory_mode in CPU_FALLBACK_MEMORY_MODES:
                # stacklevel=5 points the warning at the user's Composer(...):
                # user -> __init__ -> _init_hardware -> _fall_back_to_cpu_memory_mode -> helper
                self._fall_back_to_cpu_memory_mode(stacklevel=5)

    def _fall_back_to_cpu_memory_mode(self, stacklevel, warning=MEMORY_MODE_FALLBACK_WARNING):
        """
        Without a CUDA GPU, switch a GPU memory mode to the CPU memory mode with the same data layout ('GPU' to 'CPU',
        'SparseGPU' to 'SparseCPU'), with a warning, and set compile_model to False, since the model is only compiled on a
        GPU. stacklevel is passed to warnings.warn, so the warning points at the user's call; warning is the text of the
        warning (see _get_usable_memory_mode).
        """
        self.memory_mode = self._get_usable_memory_mode(self.memory_mode, stacklevel=stacklevel, warning=warning)
        self.compile_model = False

    @staticmethod
    def _canonical_memory_mode(memory_mode):
        """
        Return the canonical spelling of a memory mode given in any case ('cpu' gives 'CPU'), which the rest of the code
        compares against. An unknown memory mode raises here, as it would otherwise only fail when the data is prepared.
        """
        memory_modes = ('GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed')
        canonical_names = {mode.lower(): mode for mode in memory_modes}
        if not isinstance(memory_mode, str) or memory_mode.lower() not in canonical_names:
            raise ValueError(f"memory_mode must be one of {memory_modes} (in any case), not '{memory_mode}'")
        return canonical_names[memory_mode.lower()]

    @staticmethod
    def _get_usable_memory_mode(memory_mode, stacklevel, warning=MEMORY_MODE_FALLBACK_WARNING):
        """
        Return the memory mode to use on this machine for a canonical memory mode. Without a CUDA GPU, a GPU memory mode
        falls back to the CPU memory mode with the same data layout ('GPU' to 'CPU', 'SparseGPU' to 'SparseCPU'), with a
        warning; otherwise the memory mode is returned unchanged. stacklevel is passed to warnings.warn, so the warning
        points at the user's call. warning is the text of the warning, with {memory_mode} and {fallback_memory_mode} filled
        in; the default suits a memory_mode argument the user passed.
        """
        if torch.cuda.is_available() or memory_mode not in CPU_FALLBACK_MEMORY_MODES:
            return memory_mode
        fallback_memory_mode = CPU_FALLBACK_MEMORY_MODES[memory_mode]
        warnings.warn(
            warning.format(memory_mode=memory_mode, fallback_memory_mode=fallback_memory_mode), stacklevel=stacklevel)
        return fallback_memory_mode

    @staticmethod
    def _get_usable_device(device, stacklevel):
        """
        Return the device to use on this machine, as a string such as 'cpu', 'cuda' or 'cuda:1' (str(torch.device(device))).
        None gives 'cuda' when a CUDA GPU is available and 'cpu' otherwise. Without a CUDA GPU, a CUDA device falls back to
        the CPU with a warning. stacklevel is passed to warnings.warn, so the warning points at the user's call.
        """
        if device is None:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            warnings.warn(
                f"device='{device}' needs a CUDA GPU, but none is available: using device='cpu' instead.",
                stacklevel=stacklevel)
            return 'cpu'
        return str(device)

    @staticmethod
    def _canonicalize_device(device):
        """
        Return the device as a torch.device that compares equal to the same device given another way: a CUDA device
        without an index ('cuda') is the current CUDA device, so 'cuda' and 'cuda:0' are the same device on most machines.
        """
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None and torch.cuda.is_available():
            return torch.device('cuda', torch.cuda.current_device())
        return device

    def _init_data_config(self, params):
        if isinstance(params['adata'], (list, tuple)):
            assert len(params['adata']) > 0, 'adata must be an AnnData or non-empty list of AnnDatas'
            self.adata = list(params['adata'])
        else:
            self.adata = [params['adata']]
        for index, adata in enumerate(self.adata):
            self._check_adata(adata, index)
        self.categorical_covariate_keys = params['categorical_covariate_keys'] or []
        self.continuous_covariate_keys = params['continuous_covariate_keys'] or []
        self.sparse_continuous_covariate_keys = params['sparse_continuous_covariate_keys'] or []
        self.unlabeled = params['unlabeled']  # Set before the checks below, so the categorical checks can use it
        self._check_sparse_continuous_covariate_keys(self.memory_mode)
        for index, adata in enumerate(self.adata):
            self._check_categorical_covariates(adata, index)
            self._check_continuous_covariates(adata, index)
            self._check_sparse_continuous_covariates(adata, index)

        # Encodings
        self.obs_encoding_dict = {}
        self.obs_decoding_dict = {}
        self.sparse_continuous_covariates_dict = {}

        # Objects
        self.counterfactual_categorical_covariates = None
        self.counterfactual_covariates = None
        self.categorical_mask = None
        self.train_adataset = None
        self.train_adata_loader = None

    def _check_sparse_continuous_covariate_keys(self, memory_mode):
        """
        Raise an error if sparse continuous covariates are used in a memory mode other than SparseCPU.

        The model counts them toward its covariate size in every memory mode, but only SparseCPU datasets supply them, so
        other modes would fail at the first batch with a shape error ("mat1 and mat2 shapes cannot be multiplied").
        """
        if self.sparse_continuous_covariate_keys and memory_mode != 'SparseCPU':
            raise ValueError(
                f"sparse_continuous_covariate_keys is only supported in SparseCPU memory mode, not in {memory_mode} mode: "
                "use memory_mode='SparseCPU', or leave sparse_continuous_covariate_keys empty"
            )

    def _check_adata(self, adata, index, memory_mode=None):
        """
        Check one dataset for the memory mode (default: the Composer's), so unsupported inputs fail here rather than
        during training or inference. index labels the dataset in error messages.

        Composer takes AnnData objects only; reading files is left to the caller (e.g. the command line interface).
        Backed mode keeps counts on disk, so it takes AnnDatas read with read_h5ad(path, backed='r'), whose obs and var
        are in memory. The other memory modes need the counts in memory, so backed AnnDatas are refused rather than
        silently loaded whole.

        An AnnData with zero cells is refused in every memory mode.

        In-memory counts are checked for NaN, infinite and negative values. Backed counts are not checked, as that would
        read the whole file; invalid backed counts fail at the first batch that holds them, with an error from torch.
        """
        if not isinstance(adata, ad.AnnData):
            raise TypeError(
                f"adata {index} is a {type(adata).__name__}, but Composer takes AnnData objects: read the file first, "
                "e.g. ad.read_h5ad(path), or ad.read_h5ad(path, backed='r') for memory_mode='backed'"
            )
        self._check_adata_has_cells(adata, index)
        if memory_mode is None:
            memory_mode = self.memory_mode
        self._check_adata_memory_mode(adata, index, memory_mode)
        if not adata.isbacked:
            self._check_count_values(adata.X, index)

    @staticmethod
    def _check_adata_has_cells(adata, index):
        """
        Raise an error if adata has no cells. Without this check, a zero-cell training AnnData fails later with a message
        about batches (when it is the only one) or with a RuntimeError from torch's one_hot during training, and a
        zero-cell query fails with the same RuntimeError. Reads only n_obs, which backed AnnDatas also provide.
        """
        if adata.n_obs == 0:
            raise ValueError(
                f"adata {index} has no cells (n_obs is 0): remove it, or pass AnnDatas with at least one cell "
                "(e.g. check the filter or subset that produced it)"
            )

    @staticmethod
    def _check_adata_memory_mode(adata, index, memory_mode):
        """
        Raise an error if memory_mode cannot use adata: backed mode needs a backed AnnData, the other modes an in-memory
        one. Reads no counts, so it is cheap enough to check the training AnnDatas again for another memory mode.
        """
        if memory_mode == 'backed' and not adata.isbacked:
            raise ValueError(
                f"adata {index} is in memory, but backed mode keeps counts on disk: read it with ad.read_h5ad(path, backed='r'), "
                "or use memory_mode 'GPU', 'SparseGPU', 'CPU' or 'SparseCPU'"
            )
        if memory_mode != 'backed' and adata.isbacked:
            raise ValueError(
                f"adata {index} keeps its counts on disk, which memory_mode='{memory_mode}' cannot use: "
                "use memory_mode='backed', or load it into memory first (e.g. adata.to_memory())"
            )

    @staticmethod
    def _check_count_values(X, index):
        """
        Raise an error if counts hold NaN, infinite or negative values, which the count likelihood cannot model.

        Without this check they fail at the first batch with an error from torch about the encoder's Normal distribution,
        or, where sparse memory modes store counts as uint16, are silently turned into other counts (NaN to 0, -1 to 65535).
        """
        values = X.data if issparse(X) else np.asarray(X)  # Sparse matrices: only the stored values; zeros are valid
        if values.size == 0:
            return

        # NaN propagates through min and max, so two passes over the values find all three cases without copying them
        lowest, highest = values.min(), values.max()
        if np.isnan(lowest) or lowest < 0 or np.isinf(highest):
            n_nan, n_inf, n_negative = np.isnan(values).sum(), np.isinf(values).sum(), (values < 0).sum()
            raise ValueError(
                f"adata {index}.X has {n_nan} NaN, {n_inf} infinite and {n_negative} negative values, but piano models counts, "
                "which must be finite and non-negative: check that adata.X holds raw counts, not normalized, log-transformed or scaled data"
            )

    def _check_categorical_covariates(self, adata, index):
        """
        Raise an error if a categorical_covariate_keys column is missing from adata.obs or mixes kinds of values, and
        warn about columns that mix int and float values or whose float values are not whole numbers.

        Without this check a missing column fails only later: training data in run_pipeline with a bare AssertionError,
        query data with a bare KeyError from pandas. A column must hold one kind of value (number, str, bool, or one other
        Python type such as bytes); missing values (NaN, None, pd.NA, NaT) and the unlabeled value (compared by str(), as
        the encoding drops it) do not count as a kind, so [1, 2, 'Unknown'] is accepted.
        """
        for column in self.categorical_covariate_keys:
            self._check_obs_has_column(adata, index, column, 'categorical_covariate_keys')
            values = adata.obs[column]
            value_types = self._get_categorical_covariate_value_types(values, self.unlabeled)
            self._check_categorical_covariate_kinds(value_types, index, column)
            self._warn_mixed_int_float_categorical_covariate(value_types, index, column)
            self._warn_non_whole_float_categorical_covariate(values, index, column)

    @staticmethod
    def _get_categorical_covariate_value_types(values, unlabeled):
        """
        The Python types of the values of a categorical covariate column, each with one example value, in order of first
        appearance. Missing values and values whose str() is the unlabeled value are left out. For a pd.Categorical,
        only the categories the cells use are looked at, since unused categories do not reach the encoding.

        Columns whose dtype (or categories' dtype) is not object hold one kind of value, and give no types.
        """
        if isinstance(values.dtype, pd.CategoricalDtype):
            codes = values.cat.codes.to_numpy()
            values = values.cat.categories[np.unique(codes[codes >= 0])]  # Codes of missing values are -1
        if values.dtype != object:
            return {}

        # The type of every cell, not of unique() values: unique() hides True and 1.0 behind an equal 1
        cells = values.dropna().to_numpy()
        type_codes, unique_types = pd.factorize(np.array([type(value) for value in cells], dtype=object))
        value_types = {}
        for code, value_type in enumerate(unique_types):
            # Only the distinct values of one type are converted with str(), to leave out the unlabeled value
            labeled = [value for value in pd.unique(cells[type_codes == code]) if str(value) != unlabeled]
            if labeled:
                value_types[value_type] = labeled[0]
        return value_types

    @staticmethod
    def _get_categorical_covariate_kind(value_type):
        """
        The kind of value a Python type holds in a categorical covariate: 'bool', 'number', 'str', or the type's name.

        bool is tested first, since Python's bool is also a number.
        """
        if issubclass(value_type, (bool, np.bool_)):
            return 'bool'
        if issubclass(value_type, numbers.Number):
            return 'number'
        if issubclass(value_type, str):  # np.str_ is a subclass of str
            return 'str'
        return value_type.__name__

    @staticmethod
    def _check_categorical_covariate_kinds(value_types, index, column):
        """
        Raise an error if a categorical covariate column holds more than one kind of value (see
        _get_categorical_covariate_kind), given its value types from _get_categorical_covariate_value_types.

        Without this check such a column is used silently, but its categories depend on the types: values that look
        alike share a label (1 and '1' are both '1') and equal values merge into one category (True into 1).
        """
        examples = {}
        for value_type, example in value_types.items():
            examples.setdefault(Composer._get_categorical_covariate_kind(value_type), example)
        if len(examples) > 1:
            found = ', '.join(f"{kind} (e.g. {example!r})" for kind, example in examples.items())
            raise ValueError(
                f"adata {index}.obs['{column}'] mixes value types in a categorical covariate: {found}. Values of different "
                "types can share a category when they look alike (1 and '1') or are equal (1 and True), so the column must "
                "hold one kind of value: relabel the values, or convert the column to one type, e.g. to str with "
                f"adata.obs['{column}'].astype(str) if values that look alike, such as 1 and '1', should be one category"
            )

    def _check_continuous_covariates(self, adata, index):
        """
        Raise an error if a continuous_covariate_keys column of adata.obs cannot be used by the model.

        Sparse continuous covariates are a separate covariate type and are not checked here.
        """
        for column in self.continuous_covariate_keys:
            self._check_obs_has_column(adata, index, column, 'continuous_covariate_keys')
            values = adata.obs[column]
            self._check_continuous_covariate_dtype(values, index, column)
            self._check_continuous_covariate_values(values, index, column)

    def _check_stratify_column(self, adata, index):
        """
        Raise an error if the stratify_column of adata.obs cannot give every cell a class label.
        """
        self._check_obs_has_column(adata, index, self.stratify_column, 'stratify_column')
        self._check_stratify_labels(adata.obs[self.stratify_column], index, self.stratify_column)

    @staticmethod
    def _check_stratify_labels(values, index, column):
        """
        Raise an error if the stratify column has missing labels (NaN, None or pd.NA).

        np.unique cannot sort str labels mixed with NaN, and fails with a bare TypeError; treating NaN as a class of its
        own would instead give it its own samples_per_class quota and silently skew the sampling.
        """
        n_nan = int(values.isna().sum())
        if n_nan > 0:
            raise ValueError(
                f"adata {index}.obs['{column}'] has {n_nan} NaN values, but stratified sampling needs a class for every cell: "
                "fill them (e.g. with an 'unknown' class) or drop those cells"
            )

    @staticmethod
    def _check_obs_has_column(adata, index, column, argument):
        """
        Raise an error if adata.obs has no column named column, which the argument names.

        Without this check a missing column fails later with a bare KeyError from pandas, naming neither the AnnData nor
        the argument.
        """
        if column not in adata.obs.columns:
            raise ValueError(
                f"adata {index}.obs has no column '{column}', which {argument} names: add the column or remove it from {argument}"
            )

    @staticmethod
    def _warn_mixed_int_float_categorical_covariate(value_types, index, column):
        """
        Warn if a categorical covariate column holds both int and float values, given its value types from
        _get_categorical_covariate_value_types.

        Both are numbers, so the column is accepted, but equal values such as 1 and 1.0 share one category, labelled as
        the value seen first. Runs after _check_categorical_covariate_kinds, so a bool here is never mixed with numbers.
        """
        # One example of each, if any: ints (np.int64 included) are Integral, floats (np.float32 included) Real only
        int_examples = [example for value_type, example in value_types.items() if issubclass(value_type, numbers.Integral)]
        float_examples = [
            example for value_type, example in value_types.items()
            if issubclass(value_type, numbers.Real) and not issubclass(value_type, numbers.Integral)
        ]
        if int_examples and float_examples:
            warnings.warn(
                f"adata {index}.obs['{column}'] is a categorical covariate that mixes int and float values (e.g. "
                f"{int_examples[0]!r} and {float_examples[0]!r}): equal values such as 1 and 1.0 share one category, "
                "labelled as the value seen first; convert the column to one type (e.g. all int, or all str) to make the "
                "labels explicit",
                UserWarning, stacklevel=2,
            )

    @staticmethod
    def _warn_non_whole_float_categorical_covariate(values, index, column):
        """
        Warn if a float categorical covariate column (or a categorical with float categories) holds values that are not
        whole numbers.

        Such a column usually measures a quantity, but each distinct value is still used as its own category. Whole-number
        floats (e.g. 1.0, 2.0) are often integer labels stored as floats and give no warning, nor do missing values.
        """
        dtype = values.cat.categories.dtype if isinstance(values.dtype, pd.CategoricalDtype) else values.dtype
        if not pd.api.types.is_float_dtype(dtype):
            return
        distinct = np.unique(values.dropna().to_numpy(dtype=np.float64))
        if (distinct != np.floor(distinct)).any():
            warnings.warn(
                f"adata {index}.obs['{column}'] is used as a categorical covariate but holds {len(distinct)} distinct float "
                "values that are not all whole numbers, each used as its own category: if it measures a quantity, pass it "
                "in continuous_covariate_keys instead",
                UserWarning, stacklevel=2,
            )

    @staticmethod
    def _check_continuous_covariate_dtype(values, index, column, categorical_alternative=True):
        """
        Raise an error if a continuous covariate column does not hold real numbers: float or int, nullable included.

        Bool and categorical columns hold categories, not measured numbers, and object and string columns fail in torch
        when the dataset is built, even when their values look like numbers.

        By default the error suggests using the column as a categorical covariate instead. Pass
        categorical_alternative=False for a column that cannot become one, such as the value column of a sparse
        continuous covariate, so the error only suggests converting it to numbers.
        """
        dtype = values.dtype
        is_bool = pd.api.types.is_bool_dtype(dtype)
        if pd.api.types.is_numeric_dtype(dtype) and not is_bool and not pd.api.types.is_complex_dtype(dtype):
            return
        if not categorical_alternative:
            fix = "convert the column to numbers (e.g. with pd.to_numeric, or .astype(float) for bool)"
        elif is_bool:
            fix = "use it as a categorical covariate (categorical_covariate_keys)"
        else:
            fix = "convert the column (e.g. with pd.to_numeric) or use it as a categorical covariate (categorical_covariate_keys)"
        raise ValueError(f"adata {index}.obs['{column}'] has dtype {dtype}, but continuous covariates must be numbers (float or int): {fix}")

    @staticmethod
    def _check_continuous_covariate_values(values, index, column):
        """
        Raise an error if a continuous covariate column holds NaN or infinite values, which the model cannot use.

        Without this check one NaN makes every loss NaN from the first epoch, and training fails with an error from torch
        about the encoder's Normal distribution, or with no best model to save.
        """
        # Covariates are used as float32, so values beyond its range count as infinite too (reported below rather than as
        # numpy's overflow warning); missing values (pd.NA) count as NaN
        with np.errstate(over='ignore'):
            float_values = values.to_numpy(dtype=np.float32, na_value=np.nan)
        n_nan, n_inf = np.isnan(float_values).sum(), np.isinf(float_values).sum()
        if n_nan > 0 or n_inf > 0:
            raise ValueError(
                f"adata {index}.obs['{column}'] has {n_nan} NaN and {n_inf} infinite values, but continuous covariates "
                "must be finite: fill or remove the cells with missing values, or drop the column from the covariate keys"
            )

    def _check_sparse_continuous_covariates(self, adata, index):
        """
        Raise an error if a sparse_continuous_covariate_keys pair of adata.obs cannot be used by the model.

        Both the category and the value column must exist, and the value column must hold real numbers (float or int,
        nullable included) that are finite in every cell, whatever its category. The category column may hold any labels:
        its values (NaN or unseen categories) are not checked here; they are SCRUM-34's.
        """
        for category_column, value_column in self.sparse_continuous_covariate_keys:
            self._check_obs_has_column(adata, index, category_column, 'sparse_continuous_covariate_keys')
            self._check_obs_has_column(adata, index, value_column, 'sparse_continuous_covariate_keys')
            # The values are used as float32 like continuous covariates, so they share the dtype and the NaN and infinity
            # checks, dtype first as for continuous covariates. The value column cannot become a categorical covariate, so
            # the dtype error does not suggest it
            values = adata.obs[value_column]
            self._check_continuous_covariate_dtype(values, index, value_column, categorical_alternative=False)
            self._check_continuous_covariate_values(values, index, value_column)

    def _init_gene_selection(self, params):
        self.flavor = params['flavor']
        self.n_top_genes = self._canonicalize_n_top_genes(params['n_top_genes'])
        self.hvg_batch_key = params['hvg_batch_key']
        self.geneset_path = params['geneset_path']
        # Read the gene set now, so a wrong path or gene names fail (or warn) at creation rather than at initialize_features
        self.gene_set_genes = None if self.geneset_path is None else read_gene_set_genes(self.adata[0].var_names, self.geneset_path)
        if self.gene_set_genes is None and self.n_top_genes is None:
            self.n_top_genes = DEFAULT_N_TOP_GENES  # Without a gene set, None means piano's default number of HVGs
        if self.gene_set_genes is not None:
            _warn_n_top_genes_above_gene_set(self.n_top_genes, len(self.gene_set_genes))
        self.copy_adata = params['copy_adata']
        self.var_names = None

    @staticmethod
    def _canonicalize_n_top_genes(n_top_genes):
        """
        Return n_top_genes as None or an int, raising for anything else; numpy integers and floats with an integer
        value (e.g. 4096.0) are accepted. bool is rejected, although Python treats it as an integer.
        """
        if n_top_genes is None:
            return None
        is_bool = isinstance(n_top_genes, (bool, np.bool_))
        # numbers.Real covers Python and numpy integers and floats; NaN and infinity are not integer-valued
        if not is_bool and isinstance(n_top_genes, numbers.Real) and float(n_top_genes).is_integer():
            return int(n_top_genes)
        raise ValueError(
            f"n_top_genes must be None or an integer, not {n_top_genes!r}: give a number of highly variable genes, "
            "or None for the default (the whole gene set with geneset_path, else 4096 HVGs)"
        )

    def _warn_shared_counts(self):
        """Warn once if the counts of any training AnnData will be shared with the Composer (see _is_sharing_counts)."""
        shared = [index for index, adata in enumerate(self.adata) if self._is_sharing_counts(adata)]
        if shared:
            warnings.warn(
                f"The Composer shares the counts (adata.X) of training adata {shared} with your AnnData instead of copying "
                "them, to save memory: do not change adata.X in place while using the Composer, or pass copy_adata=True "
                "for a full copy",
                UserWarning, stacklevel=3,
            )

    def _init_model_config(self, params):
        self.model_kwargs = {
            # Architecture
            'n_hidden': params['n_hidden'],
            'n_layers': params['n_layers'],
            'latent_size': params['latent_size'],

            # Training mode
            'adversarial': params['adversarial'],

            # Hyperparameters
            'dropout_rate': params['dropout_rate'],
            'batchnorm_eps': params['batchnorm_eps'],
            'batchnorm_momentum': params['batchnorm_momentum'],
            'epsilon': params['epsilon'],
        }
        self.distribution = params['distribution'].lower()
        if self.distribution not in ('nb', 'zinb'):
            raise NotImplementedError('ERROR: Only NB and ZINB distributions are currently supported')
        self.model_kwargs['distribution'] = self.distribution
        self.parameterization = self._canonical_parameterization(params['parameterization'])

        # Objects
        self.model = None
        self.checkpoint_path = None

    @staticmethod
    def _canonical_parameterization(parameterization):
        """
        Return the canonical (lowercase) spelling of a parameterization given in any case ('Mu-Theta' gives 'mu-theta').
        A parameterization without a model raises here, rather than training a different model than requested.
        """
        parameterizations = ('ksi-psi', 'mu-theta')
        if not isinstance(parameterization, str) or parameterization.lower() not in parameterizations:
            raise ValueError(f"parameterization must be one of {parameterizations} (in any case), not '{parameterization}'")
        return parameterization.lower()

    def _disable_adversarial_without_categorical_covariates(self):
        """
        Turn adversarial training off, with a warning, if there are no categorical covariates, where it has no effect.

        The adversary predicts the categorical covariates; without any, it would have no outputs and add no adversarial
        loss, but its hidden layers would still run in every training step.
        """
        if self.model_kwargs['adversarial'] and not self.categorical_covariate_keys:
            print("Warning: adversarial=True has no effect without categorical_covariate_keys, so adversarial training is turned off", flush=True)
            self.model_kwargs['adversarial'] = False

    def _init_training_config(self, params):
        self.stratify_column = params['stratify_column']
        self.samples_per_class = params['samples_per_class']
        # samples_per_class turns on stratified sampling, which needs class labels from stratify_column.
        # Raise now rather than when training starts, after data and model preparation.
        if self.samples_per_class is not None and self.stratify_column is None:
            raise ValueError("samples_per_class requires stratify_column; set stratify_column or leave samples_per_class as None")
        # The training datasets read class labels from stratify_column even when samples_per_class is None,
        # so a missing column or missing labels are reported now, naming the AnnData, instead of as a bare
        # KeyError or TypeError later.
        if self.stratify_column is not None:
            for index, adata in enumerate(self.adata):
                self._check_stratify_column(adata, index)
        if self.stratify_column is not None and self.samples_per_class is None:
            warnings.warn(f"stratify_column '{self.stratify_column}' is set but samples_per_class is None, so training uses all cells without stratification")
        self.max_epochs = params['max_epochs']
        self._check_max_epochs(self.max_epochs)

        # Beta annealing
        self.batch_size = params['batch_size']
        self.min_kld_weight = params['min_kld_weight']
        self.max_kld_weight = params['max_kld_weight']
        self.min_adv_weight = params['min_adv_weight']
        self.max_adv_weight = params['max_adv_weight']
        self.n_annealing_epochs = params['n_annealing_epochs']
        if self.n_annealing_epochs is None:
            self.n_annealing_epochs = self.max_epochs
        self._check_n_annealing_epochs(self.n_annealing_epochs)

        # Hyperparameters
        self.lr = params['lr']
        self.weight_decay = params['weight_decay']
        self.shuffle = params['shuffle']
        self.drop_last = params['drop_last']
        self._check_epoch_has_batches()
        self.num_workers = params['num_workers']
        self._check_num_workers(self.num_workers, self.memory_mode)

        # Early stopping
        self.early_stopping = params['early_stopping']
        self.min_delta = params['min_delta']
        self.patience = params['patience']

        # Checkpoints
        self.save_initial_weights = params['save_initial_weights']
        self.checkpoint_every_n_epochs = params['checkpoint_every_n_epochs']
        self._check_checkpoint_every_n_epochs(self.checkpoint_every_n_epochs)

        # Logging
        self.LOSS_KEYS = ('total', 'elbo', 'nll', 'kld', 'adv', 'ref')

    def _check_epoch_has_batches(self):
        """
        Raise an error if a training epoch would have no batches, e.g. fewer cells than batch_size with drop_last=True.

        Training would otherwise fail with a ZeroDivisionError in the annealing schedule, or train nothing. The count
        needs only obs, so it is checked here with the training sampler's own counting, before data preparation.
        """
        n_cells = sum(adata.n_obs for adata in self.adata)
        class_sizes = None
        if self.samples_per_class is not None:
            # Classes are matched by name across the training AnnDatas, as in the concatenated training dataset
            labels = np.concatenate([adata.obs[self.stratify_column].to_numpy() for adata in self.adata])
            class_sizes = np.unique(labels, return_counts=True)[1]
        n_cells_per_epoch = count_cells_per_epoch(n_cells, class_sizes, self.samples_per_class)
        if count_batches(n_cells_per_epoch, self.batch_size, self.drop_last) == 0:
            raise ValueError(
                f"Training would have no batches: an epoch draws {n_cells_per_epoch} cells, fewer than batch_size={self.batch_size} "
                f"with drop_last={self.drop_last}: lower batch_size or set drop_last=False (or raise samples_per_class with stratified sampling)"
            )

    @staticmethod
    def _check_max_epochs(max_epochs):
        """
        Raise an error if training would run no epochs, which leaves no trained weights to save.
        """
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be at least 1, not {max_epochs}")

    @staticmethod
    def _check_num_workers(num_workers, memory_mode):
        """
        Raise an error for DataLoader worker processes in a GPU memory mode.

        GPU and SparseGPU datasets hold CUDA tensors, which forked worker processes cannot use. memory_mode is the mode
        actually used, after any fallback to a CPU memory mode on a machine without a CUDA GPU.
        """
        if memory_mode in ('GPU', 'SparseGPU') and num_workers > 0:
            raise ValueError(
                f"num_workers must be 0 in memory_mode='{memory_mode}' (the data is held on the GPU and worker processes "
                f"cannot use CUDA), not {num_workers}: use num_workers=0, or memory_mode='CPU', 'SparseCPU' or 'backed'"
            )

    @staticmethod
    def _check_checkpoint_every_n_epochs(checkpoint_every_n_epochs):
        """
        Raise an error for a checkpoint interval below 1 epoch; None turns periodic checkpoints off.
        """
        if checkpoint_every_n_epochs is not None and checkpoint_every_n_epochs < 1:
            raise ValueError(
                f"checkpoint_every_n_epochs must be at least 1, not {checkpoint_every_n_epochs}: "
                "use None to save no periodic checkpoints"
            )

    @staticmethod
    def _check_n_annealing_epochs(n_annealing_epochs):
        """
        Raise an error for a negative number of annealing epochs; 0 means no annealing (maximum weights from the start).
        """
        if n_annealing_epochs < 0:
            raise ValueError(f"n_annealing_epochs must be 0 (no annealing) or more, not {n_annealing_epochs}")

    def _canonicalize_early_stopping_metric(self, early_stopping_metric):
        """
        Return the canonical (lowercase) name of an early stopping metric given in any case ('REF' gives 'ref').
        An unknown name raises here, before training, rather than after a whole epoch.
        """
        if not isinstance(early_stopping_metric, str) or early_stopping_metric.lower() not in self.LOSS_KEYS:
            raise ValueError(f"early_stopping_metric must be one of {self.LOSS_KEYS} (in any case), not '{early_stopping_metric}'")
        return early_stopping_metric.lower()

    def _init_output_config(self, params):
        self.run_name = params['run_name']
        self.outdir = params['outdir']

    def _init_random_seed(self, params):
        # Stored before set_determinism runs, which reads self.random_seed when given None: random_seed=None then means
        # no seeding (random, numpy, torch and the DataLoader generator are left to the caller's random state)
        self.random_seed = params['random_seed']

    def _state_without_data(self):
        """
            Returns a shallow copy of the Composer's attributes without the training AnnDatas and their dataset and
            loader. Used for both pickling (__getstate__) and copying (__copy__, __deepcopy__).
        """
        state = self.__dict__.copy()

        # Remove large adata objects to reduce pickle size
        for _ in ['adata', 'train_adataset', 'train_adata_loader']:
            state[_] = None
        # Copy the nested dict too: the shallow copy above shares it with the live Composer
        state['_init_params'] = {**state['_init_params'], 'adata': None}

        # Model weights are already and only saved in model_checkpoint.pt
        # state['model'] = None  # Not removing model to improve user convenience

        return state

    def __getstate__(self):
        state = self._state_without_data()

        # Pickle a CPU copy of the model, so the pickle loads without a GPU even with a plain pickle.load. The model is
        # the only CUDA state: the optimizer and the compiled train step are locals of train, and only the train step is
        # compiled, never the model, so the deepcopy is safe. It briefly holds a second copy of the model on the GPU,
        # which is fine for piano's small VAE. The live model is untouched and stays on its device. Copies of a Composer
        # do not come here (see __copy__ and __deepcopy__), so they keep the model on its device.
        if state['model'] is not None:
            state['model'] = copy.deepcopy(state['model']).to('cpu')
            state['model'].device = 'cpu'

        return state

    # Copies drop the training AnnDatas, as pickling does, but keep the model on its device: only a pickle needs a
    # CPU model, and a CPU model on a copy would no longer match the copy's device and memory mode
    def __copy__(self):
        copied = type(self).__new__(type(self))
        copied.__dict__.update(self._state_without_data())
        return copied

    def __deepcopy__(self, memo):
        copied = type(self).__new__(type(self))
        # Register the copy first, as copy.deepcopy does, so references back to this Composer point at the copy
        memo[id(self)] = copied
        copied.__dict__.update(copy.deepcopy(self._state_without_data(), memo))
        return copied

    def save(self, path):
        """
            Pickles the Composer without its training AnnDatas and their dataset and loader. The pickle holds a CPU
            copy of the model, so it loads without a GPU; the Composer's own model stays on its device.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path, model_checkpoint_path=None, device=None):
        """
            This static method loads the Composer from a pickle file and sets the device. The default device is 'cuda'
            when a CUDA GPU is available and 'cpu' otherwise; a CUDA device asked for without a GPU falls back to 'cpu',
            with a warning.
            If a model checkpoint path is specified, prepares model from pianist.model_kwargs and loads weights.
            The weights are loaded from checkpoint using load_model, which uses the device specified and sets model to eval.
            Without a checkpoint, the pickled model is moved to the device and set to eval mode in the same way.
            The pickled tensors are loaded to the device, so a Composer saved with its model on a GPU loads on a machine
            without one.
            Without a CUDA GPU, a Composer saved in memory_mode 'GPU' or 'SparseGPU' falls back to 'CPU' or 'SparseCPU'
            (and compile_model to False), with a warning, as at Composer creation. On a GPU node the memory mode is kept,
            also with device='cpu': a GPU memory mode then builds batches on the GPU and moves them to the model.
        """
        # stacklevel=3 points the warning at the user's Composer.load(...): user -> load -> helper
        device = Composer._get_usable_device(device, stacklevel=3)
        # Unpickle the tensors straight to the device, so a Composer pickled with its model on a GPU loads without one
        with open(path, 'rb') as f:
            pianist = _DeviceMappingUnpickler(f, device).load()
        if not torch.cuda.is_available() and pianist.memory_mode in CPU_FALLBACK_MEMORY_MODES:
            # stacklevel=4 points the warning at the user's Composer.load(...):
            # user -> load -> _fall_back_to_cpu_memory_mode -> helper
            pianist._fall_back_to_cpu_memory_mode(stacklevel=4, warning=LOAD_MEMORY_MODE_FALLBACK_WARNING)
            # The pickled dataset builder was made for the GPU memory mode
            pianist._set_adataset_builder()
        # Set before load_model, so that it does not warn about moving the model to another device
        pianist.device = device

        if model_checkpoint_path is not None:
            pianist.prepare_model(**pianist.model_kwargs)
            pianist.load_model(model_checkpoint_path, device=device)
        elif pianist.model is not None:
            # The model was pickled where it was trained (e.g. on cuda) and in training mode
            pianist._prepare_model_for_inference(device)

        return pianist

    def load_model(self, model_checkpoint_path, device=None):
        """
            Loads the model weights from specified checkpoint to specified device, by default the Composer's device.
            A CUDA device asked for without a GPU falls back to 'cpu', with a warning. Passing a device other than
            the Composer's moves the Composer there too, with a warning.
            This also sets the model to eval mode. To continue training, set .train()
            Loaded weights count as trained, so get_latent_representation and get_counterfactual do not warn about
            random weights. A checkpoint holds only the weights, so this also applies to model_epoch=-1.pt, the initial
            random weights saved with save_initial_weights or checkpoint_every_n_epochs.
        """
        # stacklevel=3 points the warning at the user's load_model(...): user -> load_model -> helper
        requested_device = device
        device = self._get_usable_device(self.device if device is None else device, stacklevel=3)
        # Only a device the user passed gets this warning: with device empty, the model moves only when the Composer's
        # device needs a GPU that is missing, which the fallback warning already explains
        if requested_device is not None and self._canonicalize_device(device) != self._canonicalize_device(self.device):
            warnings.warn(
                f"load_model moves the model from device='{self.device}' to device='{device}'. Leave device empty to "
                f"keep the model on the Composer's device.",
                stacklevel=2)

        self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=device, weights_only=True))
        self.trained_model = True  # Loaded weights count as trained: inference does not warn about random weights
        self.device = device
        self._prepare_model_for_inference(device)

        return self.model

    def _prepare_model_for_inference(self, device):
        """
            Moves the model to the device (setting model.device too) and sets it to eval mode, after loading.
        """
        self._move_model_to_device(device)
        self.model.eval()

    def _move_model_to_device(self, device):
        """
            Moves the model to the device in place, setting model.device too, which the model uses for its batches.
        """
        self.model.device = device
        self.model.to(device)

    # ======================
    # Porcelain: Public API
    # ======================
    def run_pipeline(self):
        self.initialize_features()
        self.prepare_data()
        self.prepare_model(**self.model_kwargs)
        self.train()

    def get_latent_representation(
        self,
        adata=None,
        memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'] | None] = None,
        batch_size: int = 4096,
        mc_samples: int = 0,
    ):
        """
        Latent representation of each cell of adata (the training data if None).

        :param adata: AnnData whose cells to embed, which gives exactly its own cells. None (default) gives the cells of
            all training AnnDatas, concatenated in the order they were passed to the Composer, in any memory_mode. None is
            not available after Composer.save/load or on a copy of a Composer, which do not keep the training AnnDatas.
        :param memory_mode: Memory mode for the data. Default (None) uses the Composer's memory mode. Without a CUDA GPU,
            'GPU' and 'SparseGPU' fall back to 'CPU' and 'SparseCPU' with a warning. 'GPU' and 'SparseGPU' use 0 DataLoader
            workers for this call, with a warning if the Composer has num_workers > 0.
        :param mc_samples: 0 (default) returns each cell's posterior mean, which is deterministic. A positive number returns
            the average of that many random draws from each cell's posterior; mc_samples=1 gives one draw, which differs from
            call to call by about the posterior's standard deviation.
        :raises ValueError: If adata has no cells, or if a categorical covariate column of adata is missing, or a
            continuous covariate column is missing, not numeric or not finite, or a sparse continuous covariate column
            is missing, or its value column is not numeric or not finite, although the latent representation does not
            use covariates (fill unknown values with any finite placeholder, e.g. 0.0), or if adata is None on a
            Composer rebuilt by Composer.load or copied, or if the Composer has no model (model is None: prepare_model(),
            train() or Composer.load never built one).

        Warns (UserWarning), once per column, if a sparse continuous covariate category column of adata has values not
        seen in training: those cells are encoded as untreated, which the latent representation does not use.

        Warns (UserWarning) if the model was neither trained nor loaded, since the latent representation then comes from
        random weights.
        """
        self._check_model_is_ready()  # First: nothing below can succeed without a model
        self._warn_untrained_model()
        self._check_mc_samples(mc_samples)
        memory_mode = self._get_inference_memory_mode(memory_mode)
        if adata is not None:
            self._warn_unseen_sparse_categories(
                self._get_unseen_sparse_categories(adata),
                "their cells are encoded as zeros in every column of this covariate, i.e. as untreated; the latent "
                "representation does not use covariates, so it is not affected",
            )

        adataset = self._get_adataset(adata, memory_mode)
        adata_loader = DataLoader(
            adataset, batch_size=None, num_workers=self._get_num_workers(memory_mode, stacklevel=3),
            sampler=self._get_sampler(adataset, batch_size=batch_size, samples_per_class=None, shuffle=False, drop_last=False, memory_mode=memory_mode),
        )
        latent_space = self.model.get_latent_representation(
            adata_loader, mc_samples=mc_samples,
        )
        print(f'Retrieving latent space with shape {latent_space.shape}', flush=True)

        return latent_space

    def _get_inference_memory_mode(self, memory_mode):
        """
        Return the memory mode to use for the memory_mode passed to get_latent_representation or get_counterfactual:
        None gives the Composer's memory mode, which was already resolved for this machine when the Composer was created or
        loaded; any other value is canonicalized and, without a CUDA GPU, falls back as at Composer creation (see
        _get_usable_memory_mode).
        """
        if memory_mode is None:
            return self.memory_mode
        # stacklevel=4 points the fallback warning at the user's call: user -> get_* -> this method -> helper
        return self._get_usable_memory_mode(self._canonical_memory_mode(memory_mode), stacklevel=4)

    def _check_model_is_ready(self):
        """
        Raise if this Composer has no model, so get_latent_representation and get_counterfactual fail before any feature
        initialization or dataset building. The model is None until prepare_model(), and Composer.load keeps it None for
        a Composer that was saved before training.
        """
        if self.model is None:
            raise ValueError(
                "This Composer has no model: train one with run_pipeline() (or initialize_features(), prepare_data(), "
                "prepare_model() and train()), or load a trained one with Composer.load(path, model_checkpoint_path=...)"
            )

    def _warn_untrained_model(self):
        """
        Warn if the model was built (prepare_model) but neither trained nor loaded, so its output comes from random
        weights. train() and load_model() (also through Composer.load with a checkpoint) set trained_model, and a
        Composer pickled after training keeps it. A checkpoint file holds only the weights and does not record whether
        they were trained, so weights loaded from any checkpoint count as trained, including model_epoch=-1.pt (the
        initial random weights saved with save_initial_weights or checkpoint_every_n_epochs).
        """
        if not self.trained_model:
            # stacklevel=3 points the warning at the user's call: user -> get_* -> this method
            warnings.warn(
                "This Composer's model was neither trained nor loaded, so its output comes from random weights: train it "
                "with train() (or run_pipeline()), or load trained weights with load_model(model_checkpoint_path) or "
                "Composer.load(path, model_checkpoint_path=...)",
                stacklevel=3)

    @staticmethod
    def _check_mc_samples(mc_samples):
        """Raise unless mc_samples is 0 (use each cell's posterior mean) or a positive number of posterior draws."""
        if mc_samples < 0:
            raise ValueError(f"mc_samples must be 0 (posterior mean) or a positive number of posterior draws, not {mc_samples}")

    def _get_unseen_sparse_categories(self, adata):
        """
        Return {category column: {unseen value: number of cells}} for the sparse continuous covariate category columns
        of adata.obs that have values not seen in training, which the SparseCPU dataset encodes as zeros (untreated).

        :raises ValueError: If adata.obs has no category column of sparse_continuous_covariate_keys.
        """
        unseen = {}
        for (category_column, _) in self.sparse_continuous_covariate_keys:
            # This runs before _get_adataset checks the query, so it raises the same missing-column error itself
            self._check_obs_has_column(adata, 'query', category_column, 'sparse_continuous_covariate_keys')
            _, encoding_dict = self.sparse_continuous_covariates_dict[category_column]
            unseen_values = get_unseen_values(adata.obs[category_column], list(encoding_dict))
            if unseen_values:
                unseen[category_column] = unseen_values
        return unseen

    @staticmethod
    def _format_unseen_values(column, unseen_values):
        """Describe the unseen values of one column with their numbers of cells, values sorted by str()."""
        descriptions = []
        for value, n_cells in sorted(unseen_values.items(), key=lambda item: str(item[0])):
            if pd.isna(value):
                value_text = 'NaN (missing)'
            else:
                # Numpy scalars are shown as Python values, e.g. 3 rather than np.int64(3)
                value_text = repr(value.item() if isinstance(value, np.generic) else value)
            descriptions.append(f"{value_text}: {n_cells} cells")
        return f"column '{column}' of sparse_continuous_covariate_keys has values not seen in training ({', '.join(descriptions)})"

    def _warn_unseen_sparse_categories(self, unseen, consequence):
        """Warn once per column of unseen (from _get_unseen_sparse_categories); consequence says what this means for the caller."""
        for column, unseen_values in unseen.items():
            warnings.warn(
                f"adata.obs {self._format_unseen_values(column, unseen_values)}: {consequence}",
                UserWarning, stacklevel=3,
            )

    def get_counterfactual(
        self,
        adata=None,
        covariates: Union[Literal['marginal'], dict, None] = 'marginal',
        batch_size: int = 4096,
        memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'] | None] = None,
        mc_samples: int = 0,
    ):
        """
        Retrieve counterfactual representations of passed in data.
        This is useful for getting batch corrected counts, reconstructions, or simulating effects of alternate covariate values.
        Returns each cell's expected counts: the NB mean, times one minus the dropout probability for distribution='zinb'.

        :param adata: AnnData to retrieve counterfactual representations, which gives exactly its own cells. None (default)
            gives the cells of all training AnnDatas, concatenated in the order they were passed to the Composer, in any
            memory_mode. None is not available after Composer.save/load or on a copy of a Composer, which do not keep the
            training AnnDatas.
        :param covariates: 
            - marginal (default):
                Compute with categorical covariates averaged per covariate key and continuous covariates using original values
            - dict: {key: value | [value1, value2, ...] | 'marginal'}
                For a given categorical covariate key, there are three options:
                    - One value can be specified (treated as one-hot encoded where that value is set to 1, and all others set to 0).
                      The unlabeled value (e.g. 'Unknown') sets all values to 0, as for unlabeled cells in training.
                    - Multiple can be specified (each value specified set to 1/k, for k total specified, and all others set to 0).
                    - 'marginal', where every possible value is set to 1/n, for all n total possible covariate values for the given covariate.
                    Values are matched to the training labels as strings, so a column of integers, floats or bools can be
                    given 1, 1.0 or True as well as '1', '1.0' or 'True'.
                For a given continuous covariate key, one value can be specified to replace the original covariate value.
                For a given sparse continuous covariate key, dictionary of category:value mappings or None can be specified.
                    For each category, the corresponding value will be assigned to that category in the encoding.
                    If an empty dict or None is passed in, all values for the sparse continuous covariate key will be set to 0.
                    Example 1:
                        'drug': {'A': 0.05}, # Assigns value for one drug, other drugs set to 0
                    Example 2:
                        'drug': {'A': 0.05, 'B': 0.5}, # Assigns values for multiple drugs, unassigned are set to 0
                    Example 3:
                        'drug': {}, # Assigns 0.0s for all drugs
                    Example 4:
                        'drug': None, # Assigns 0.0s for all drugs
                    Sparse continuous covariate keys enable large datasets with sparsely encoded continuous covariates to save memory, such as if only one drug at a time is used per sample.
                    Query cells whose category was not seen in training must have that covariate replaced this way.
                If left unspecified, the original value of a covariate is used.
            - None:
                Use the actual covariates in the metadata, which provides a probabilistic reconstruction of the original data.
        :type covariates: Union[Literal['marginal'], dict, None]
        :param batch_size: Number of cells to compute at once
        :type batch_size: int
        :param memory_mode: Memory mode for sampler. Default (None) uses same memory mode as Composer. Without a CUDA GPU,
            'GPU' and 'SparseGPU' fall back to 'CPU' and 'SparseCPU' with a warning. 'GPU' and 'SparseGPU' use 0 DataLoader
            workers for this call, with a warning if the Composer has num_workers > 0.
        :type memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'] | None]
        :param mc_samples: 0 (default) decodes each cell's posterior mean, so counterfactuals are deterministic. A positive
            number averages the expected counts decoded from that many random draws from each cell's posterior (one draw
            for 1).
        :type mc_samples: int
        :raises ValueError: If covariates is not one of the above, or names a key that is not a covariate of this Composer,
            or gives a categorical value that is not a training label; if mc_samples is negative; if adata is None on a
            Composer rebuilt by Composer.load or copied, or has no cells; if a categorical covariate column of adata is
            missing; if a continuous covariate column is missing, not numeric or not finite; if a sparse continuous
            covariate column is missing, or its value column is not numeric or not finite; or if a sparse continuous
            covariate category column has values not seen in training and covariates keeps that covariate (None,
            'marginal', or a dict without that category column; a dict that replaces it only warns); or if the Composer
            has no model (model is None: prepare_model(), train() or Composer.load never built one).

        Warns (UserWarning) if the model was neither trained nor loaded, since the counterfactuals then come from random
        weights.
        """
        self._check_model_is_ready()  # First: nothing below can succeed without a model
        self._warn_untrained_model()
        self._check_counterfactual_covariates(covariates)  # Before loading data, so typos fail fast
        self._check_mc_samples(mc_samples)
        memory_mode = self._get_inference_memory_mode(memory_mode)
        self._check_sparse_continuous_covariate_keys(memory_mode)  # The decoder needs the covariates the model was trained with
        if adata is not None:
            unseen = self._get_unseen_sparse_categories(adata)
            self._check_unseen_sparse_categories_not_used(unseen, covariates)  # Raises before any warning
            self._warn_unseen_sparse_categories(
                unseen, "their original category is replaced by the counterfactual values in covariates, so it is not used",
            )

        adataset = self._get_adataset(adata, memory_mode)
        adata_loader = DataLoader(
            adataset, batch_size=None, num_workers=self._get_num_workers(memory_mode, stacklevel=3),
            sampler=self._get_sampler(adataset, batch_size=batch_size, samples_per_class=None, shuffle=False, drop_last=False, memory_mode=memory_mode),
        )
        mask, counterfactuals = None, None  # Get reconstruction if covariates set to None
        if covariates == 'marginal':
            mask, counterfactuals = self.categorical_mask, self.counterfactual_covariates
        elif isinstance(covariates, dict):
            mask, counterfactuals = self._encode_counterfactual_covariates(covariates)
        counterfactual_counts = self.model.get_counterfactuals(
            adata_loader, mask=mask, counterfactuals=counterfactuals, mc_samples=mc_samples,
        )
        print(f'Retrieving counterfactual gene counts with shape {counterfactual_counts.shape}', flush=True)

        return counterfactual_counts

    def _check_unseen_sparse_categories_not_used(self, unseen, covariates):
        """
        Raise if a column of unseen (from _get_unseen_sparse_categories) keeps the cells' original covariate: covariates
        is None, 'marginal' (which averages categorical covariates only) or a dict without that column, as in
        _encode_counterfactual_covariates. Those cells would be decoded as untreated.
        """
        # Only a dict naming the category column replaces it; None and 'marginal' keep every sparse covariate
        replaced_columns = covariates.keys() if isinstance(covariates, dict) else ()
        problems = []
        for column, unseen_values in unseen.items():
            if column in replaced_columns:
                continue
            problems.append(
                f"adata.obs {self._format_unseen_values(column, unseen_values)}, and covariates does not replace this "
                f"covariate, so those cells would be encoded as zeros in every column of it, i.e. as untreated; pass "
                f"covariates={{'{column}': {{category: value}}}} or {{'{column}': {{}}}} to replace it, or drop or "
                f"relabel those cells"
            )
        if problems:
            raise ValueError('; '.join(problems))

    def _check_counterfactual_covariates(self, covariates):
        """
        Raise if covariates is not 'marginal', None or a dict, since anything else used to return a reconstruction silently.
        """
        if isinstance(covariates, dict):
            self._check_counterfactual_covariate_keys(covariates)
        elif not (covariates is None or (isinstance(covariates, str) and covariates == 'marginal')):
            raise ValueError(
                f"covariates must be 'marginal', None or a dict mapping covariate keys to values, got {covariates!r}"
            )

    def _check_counterfactual_covariate_keys(self, covariates_dict):
        """
        Raise if a key of covariates_dict is not a covariate of this Composer, since unknown keys used to be ignored silently.
        Sparse continuous covariates are keyed by their category column; their value column is not a valid key.
        """
        category_columns = [category_column for (category_column, _) in self.sparse_continuous_covariate_keys]
        valid_keys = list(self.categorical_covariate_keys) + list(self.continuous_covariate_keys) + category_columns
        for (category_column, value_column) in self.sparse_continuous_covariate_keys:
            if value_column in covariates_dict and value_column not in valid_keys:
                raise ValueError(
                    f"'{value_column}' is the value column of sparse continuous covariate '{category_column}'; "
                    f"give counterfactual values as {{'{category_column}': {{category: value}}}}"
                )
        unknown_keys = [key for key in covariates_dict if key not in valid_keys]
        if unknown_keys:
            raise ValueError(f"Unknown covariate keys {unknown_keys} in covariates; valid keys are {valid_keys}")

    def _encode_counterfactual_covariates(self, covariates_dict):
        """
        This function creates a mask, which is multiplied with the original covariates to select which original values to keep.
        The counterfactual covariates matrix is added to the masked original covariates to produce the final covariates matrix.
        Covariate keys not specified in the covariates_dict are not updated, in which case the original covariate values are used.
            I.e., covariates_used = original_covariates * mask + counterfactual_covariates * (1 - mask)
            Note: the current implementation sets counterfactual_covariates that are NOT changed to 0s, so multiplying by (1 - mask) is a no-op.

        For a given categorical covariate key, there are three options supported:
            - One value can be specified (treated as one-hot encoded where that value is set to 1, and all others set to 0).
              The unlabeled value (e.g. 'Unknown') sets all values to 0, as for unlabeled cells in training.
            - Multiple can be specified (each value specified set to 1/k, for k total specified, and all others set to 0).
            - 'marginal', where every possible value is set to 1/n, for all n total possible covariate values for the given covariate.
        For a given continuous covariate key, one value can be specified to replace the original covariate value.
        For a given sparse continuous covariate key, dictionary of category:value mappings or None can be specified.
            For each category, the corresponding value will be assigned to that category in the encoding.
            If an empty dict or None is passed in, all values for the sparse continuous covariate key will be set to 0.
            Example 1:
                'drug': {'A': 0.05}, # Assigns value for one drug, other drugs set to 0
            Example 2:
                'drug': {'A': 0.05, 'B': 0.5}, # Assigns values for multiple drugs, unassigned are set to 0
            Example 3:
                'drug': {}, # Assigns 0.0s for all drugs
            Example 4:
                'drug': None, # Assigns 0.0s for all drugs
            Sparse continuous covariate keys enable large datasets with sparsely encoded continuous covariates to save memory, such as if only one drug at a time is used per sample.
        """
        mask_blocks = []
        counterfactual_blocks = []

        # Create blocks for each categorical key
        for key in self.categorical_covariate_keys:
            n_categories = max(self.obs_encoding_dict[key].values()) + 1
            mask_block = np.ones(n_categories, dtype=np.float32)  # Ones due to default of keeping original value when multiplying by mask
            replacement_block = np.zeros(n_categories, dtype=np.float32)  # Zeros due to default addition to original after masking

            if key in covariates_dict:  # Update with counterfactual covariate values iff specified in covariates_dict
                mask_block[:] = 0.0  # Mask out original covariate values if replacing with counterfactuals
                replacement_block = self._encode_counterfactual_categorical(key, covariates_dict[key])

            mask_blocks.append(mask_block)
            counterfactual_blocks.append(replacement_block)

        # Create blocks for each continuous key
        for key in self.continuous_covariate_keys:
            mask_block = np.ones(1, dtype=np.float32)  # Ones due to default of keeping original value when multiplying by mask
            replacement_block = np.zeros(1, dtype=np.float32)  # Zeros due to default addition to original after masking

            if key in covariates_dict:  # Update with counterfactual covariate values iff specified in covariates_dict
                mask_block[0] = 0.0
                replacement_block[0] = covariates_dict[key]

            mask_blocks.append(mask_block)
            counterfactual_blocks.append(replacement_block)

        # Create blocks for each sparse continuous key
        for (category_column, value_column) in self.sparse_continuous_covariate_keys:
            # Implementation details: self.sparse_continuous_covariate_keys is a list of pairs (category column, value column); the dict maps the category column to (n_categories, encoding_dict)
            n_categories, encoding_dict = self.sparse_continuous_covariates_dict[category_column]
            mask_block = np.ones(n_categories, dtype=np.float32)
            replacement_block = np.zeros(n_categories, dtype=np.float32)

            if category_column in covariates_dict:  # Update with counterfactual covariate values iff specified in covariates_dict
                values_dict = covariates_dict[category_column]
                mask_block[:] = 0.0  # Mask out original covariate values if replacing with counterfactuals

                # Update counterfactual covariate block based on input value(s)
                if values_dict is None:
                    values_dict = {}
                for category in values_dict:
                    if category not in encoding_dict:
                        raise ValueError(
                            f"No encoding for sparse continuous covariate '{category}' found! Please ensure this sparse continuous covariate was encoded during training."
                        )
                    try:
                        value = float(values_dict[category])
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"Value for category '{category}' in '{category_column}' must be numeric, got {values_dict[category]}"
                        )
                    replacement_block[encoding_dict[category]] = value  # Replace original value with counterfactual
            mask_blocks.append(mask_block)
            counterfactual_blocks.append(replacement_block)

        mask = np.concatenate(mask_blocks)
        counterfactuals = np.concatenate(counterfactual_blocks)
        assert mask.shape[0] == self.n_total_covariate_dims
        assert counterfactuals.shape[0] == self.n_total_covariate_dims

        return mask, counterfactuals

    def _encode_counterfactual_categorical(self, key, value):
        """
        Encode the counterfactual value(s) of one categorical covariate as the block of covariate values used in its place.
        One value is one-hot encoded; k values are each set to 1/k; 'marginal' sets all n values to 1/n.
        The unlabeled value (e.g. 'Unknown') is encoded as all zeros, as unlabeled cells are in training.
        Values are matched to the training labels by str(value), as the labels were built, so 1, 1.0 or True match
        the labels '1', '1.0' or 'True'.
        """
        encoding_dict = self.obs_encoding_dict[key]
        n_categories = max(encoding_dict.values()) + 1
        if value == 'marginal':
            return np.full(n_categories, 1.0 / n_categories, dtype=np.float32)  # Marginalize over all n possible values

        # Treat one value as a list of one, so both share the encoding below
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            values = value
        else:
            values = [value]
        # Training labels are stored as str(value) (encode_categorical_covariates), so match values by the same string;
        # remove duplicates after the conversion (e.g. 1 and '1'), preserving order
        labels = list(dict.fromkeys(str(v) for v in values))
        k = len(labels)
        assert k > 0, f"ERROR: Cannot specify empty list of categorical covariates for covariate: {key}"

        replacement_block = np.zeros(n_categories, dtype=np.float32)
        for label in labels:
            if label not in encoding_dict:
                valid_labels = sorted(name for name, code in encoding_dict.items() if code >= 0)
                raise ValueError(
                    f"Invalid value '{label}' for covariate '{key}': use one of the training labels {valid_labels}, "
                    f"a list of them, or 'marginal'"
                )
            if encoding_dict[label] < 0:
                continue  # The unlabeled value (encoded -1) is all zeros, as for unlabeled cells in training
            replacement_block[encoding_dict[label]] = 1.0 / k  # Marginalize over k specified values (one-hot if k = 1)

        return replacement_block

    def set_determinism(self, deterministic: bool = None, random_seed: int = None):
        if random_seed is None:
            random_seed = self.random_seed
        else:
            self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

        if deterministic is None:
            deterministic = self.deterministic
        else:
            self.deterministic = deterministic

        if not deterministic:
            # These settings are process-wide, so turn off what an earlier deterministic Composer turned on
            torch.backends.cudnn.deterministic = False
            torch.use_deterministic_algorithms(False)
            return None, None

        # Deterministic cuBLAS matrix products need a fixed workspace size, set by an environment variable
        # (https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility); without it, deterministic training on a
        # GPU fails at the first matrix product. Set it unless the user chose a value. torch reads it only once, at the first
        # GPU matrix product in the process (checked with torch 2.7.1), so this works when the Composer is created before
        # any GPU work, as in a usual script, and is too late if the process already ran GPU matrix products
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            print("Note: deterministic=True sets the environment variable CUBLAS_WORKSPACE_CONFIG=':4096:8', which deterministic GPU training needs (set it yourself to choose another value)", flush=True)
            if torch.cuda.is_initialized():
                print("Warning: CUDA is already in use in this process: if it already ran GPU matrix products, deterministic GPU training fails; set CUBLAS_WORKSPACE_CONFIG=':4096:8' before starting Python, or use deterministic=False", flush=True)

        # Use deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        # Deterministic workers
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        dataloader_generator = torch.Generator()
        if random_seed is not None:
            dataloader_generator.manual_seed(random_seed)

        return seed_worker, dataloader_generator

    # ======================
    # Plumbing: Internal API
    # ======================
    def _select_genes(self):
        """
        Return the genes to train on, with the same rules in every memory mode, as a pandas Index in the first AnnData's
        gene order (not sorted), which saving the model needs (var_names.to_series()).
        With a gene set (geneset_path, read when the Composer was created): the whole set when n_top_genes is None, <= 0
        or at least the number of gene-set genes found; else the n_top_genes HVGs ranked within the set, whose
        mean-variance trend is fitted on the gene-set genes only (this can differ from ranking all genes first).
        Without a gene set: all genes of the first AnnData when n_top_genes <= 0; else the n_top_genes HVGs.
        """
        if not self._is_selecting_hvgs():
            # No HVGs are ranked, so hvg_batch_key is not used. With a gene set, n_top_genes <= 0 means the whole set
            # rather than all genes found in the first adata
            return self.adata[0].var_names.copy() if self.gene_set_genes is None else self.gene_set_genes
        if self.gene_set_genes is not None:
            return _select_hvgs_within_gene_set(self.adata, self.gene_set_genes, self.n_top_genes, batch_key=self._get_hvg_batch_key())
        return self._select_hvgs()

    def _select_hvgs(self):
        """
        Return the n_top_genes highly variable genes across all AnnDatas, in the AnnData's gene order.
        Batches are given by hvg_batch_key if the first AnnData has that column; in backed mode the counts are read from
        disk in chunks.
        """
        # inplace=False leaves the caller's AnnDatas unchanged (no subsetting, no added var columns or uns['hvg'])
        hvg = highly_variable_genes(self.adata, n_top_genes=self.n_top_genes, inplace=False, batch_key=self._get_hvg_batch_key())
        return hvg.index[hvg['highly_variable'].to_numpy()]

    def _get_hvg_batch_key(self):
        """Return hvg_batch_key if the first AnnData has that column, else None (with a message), for HVG selection."""
        if self.hvg_batch_key is not None and self.hvg_batch_key not in self.adata[0].obs:
            print(f'Unable to find hvg_batch_key {self.hvg_batch_key} in adata.obs for HVG', flush=True)
        return self.hvg_batch_key if self.hvg_batch_key in self.adata[0].obs else None

    def _check_adatas_have_genes(self, var_names):
        """
        Raise a ValueError if any training AnnData is missing some of the genes var_names. Called in initialize_features,
        right after the genes are selected, in every memory mode.
        """
        for index, adata in enumerate(self.adata):
            self._check_adata_has_genes(adata, index, var_names)

    def _check_adata_has_genes(self, adata, label, var_names):
        """
        Raise a ValueError naming adata by label (a training index or 'query') if it is missing some of the genes
        var_names. The fix hint depends on the label: a query cannot change the model's genes.
        """
        missing = pd.Index(var_names).difference(adata.var_names)
        if len(missing) > 0:
            if label == 'query':
                fix = "a query must have all of the model's genes (Composer.var_names)"
            else:
                fix = (
                    "every training AnnData must have all selected genes (subset them to the genes they share, or give "
                    "a geneset_path whose genes they all have)"
                )
            raise ValueError(
                f"adata {label} is missing {len(missing)} of the {len(var_names)} selected genes, e.g. {list(missing[:5])}: "
                f"{fix}"
            )

    def _is_selecting_hvgs(self):
        """
        Return True if the training genes are chosen by HVG selection, which needs the counts: without a gene set,
        n_top_genes > 0; with one, 0 < n_top_genes < the gene-set genes found (HVGs ranked within the set). Otherwise
        they are the whole gene set or all genes of the first AnnData, known from the AnnDatas' var_names alone.
        """
        n_gene_set_genes = None if self.gene_set_genes is None else len(self.gene_set_genes)
        return _uses_hvg_selection(self.n_top_genes, n_gene_set_genes)

    def _is_sharing_counts(self, adata):
        """
        Return True if the Composer can use the counts of training AnnData adata as they are instead of copying them.

        That is the case for an in-memory AnnData (not backed, not a view) whose genes are already exactly the training
        genes, in the same order, unless copy_adata is True.
        """
        # A view re-slices its parent on every read and follows edits to it, so it is copied (materialized)
        if self.copy_adata or self.memory_mode == 'backed' or adata.is_view:
            return False
        # This runs when the Composer is created, where the sharing warning is given; HVGs are only selected later, in
        # initialize_features, so with HVG selection (also HVGs within a gene set) the training genes are not known yet
        # and the counts are copied
        if self._is_selecting_hvgs():
            return False
        return adata.var_names.equals(self._select_genes())

    def _build_training_adata(self, adata, var_names):
        """
        Return the Composer's own AnnData for in-memory training AnnData adata, with only the genes var_names (in that
        order) and only X, obs and var: the counts are shared with adata if _is_sharing_counts, else copied.
        """
        if self._is_sharing_counts(adata):
            # Each dataset stores its own copy of the counts (to_stored_dense, to_stored_csr), so training never changes them
            X = adata.X
        else:
            adata = adata[:, var_names]  # A view with the selected genes, in their order
            X = adata.X.copy()
        # obs and var are always the Composer's own: SparseCPU datasets read obs on every batch, so caller edits must not reach them
        return ad.AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())

    def initialize_features(self):
        if self.geneset_path is None:
            print(f'Preparing data with highly_variable_genes flavor = {self.flavor}, {self.n_top_genes} HVGs (-1 = all HVGs if selected before passing into Composer), batch key = {self.hvg_batch_key}')
        else:
            # Same rule as _select_genes; the batch key matters only when HVGs are ranked within the set
            if not self._is_selecting_hvgs():
                print(f'Preparing data with gene set: {self.geneset_path} (all {len(self.gene_set_genes)} genes of the set)')
            else:
                print(f'Preparing data with gene set: {self.geneset_path} ({self.n_top_genes} HVGs within the set), batch key = {self.hvg_batch_key}')

        var_names = self._select_genes()
        # Every training AnnData must have the selected genes. Checked in every memory mode, so the message (with its
        # fix) is the same everywhere instead of anndata's bare KeyError from the gene subset in the in-memory modes;
        # backed datasets rely on it, since each reads these genes in this order (BackedAnnDataset.set_var_subset)
        self._check_adatas_have_genes(var_names)
        if self.memory_mode != 'backed':
            self.adata = [self._build_training_adata(adata, var_names) for adata in self.adata]
        self.var_names = var_names  # Safely handles both backed and non-backed modes
        self.input_size = len(self.var_names)

        # Encode covariates
        obs_list = [_.obs for _ in self.adata]
        self.counterfactual_categorical_covariates, self.obs_encoding_dict, self.obs_decoding_dict = encode_categorical_covariates(obs_list, self.categorical_covariate_keys, self.unlabeled)
        self.sparse_continuous_covariates_dict = encode_sparse_continuous_covariates(obs_list, self.sparse_continuous_covariate_keys)

        # Save number of covariate dimensions
        self.n_categorical_covariate_dims = int(np.sum([max(self.obs_encoding_dict[_].values()) + 1 for _ in self.categorical_covariate_keys]))
        self.n_continuous_covariate_dims = len(self.continuous_covariate_keys)
        # Implementation details: self.sparse_continuous_covariate_keys is a list of pairs (category column, value column); the dict maps the category column to (n_categories, encoding_dict)
        self.n_sparse_continuous_covariate_dims = int(np.sum([self.sparse_continuous_covariates_dict[_[0]][0] for _ in self.sparse_continuous_covariate_keys]))
        self.n_total_covariate_dims = int(self.n_categorical_covariate_dims + self.n_continuous_covariate_dims + self.n_sparse_continuous_covariate_dims)

        # Save counterfactual covariates
        self.counterfactual_covariates = np.pad(self.counterfactual_categorical_covariates, (0, self.n_continuous_covariate_dims + self.n_sparse_continuous_covariate_dims))  # Padding ensures the dimensions are the same, with zeros for the continuous covariates
        self.categorical_mask = np.concatenate([
            np.zeros(self.n_categorical_covariate_dims, dtype=np.float32),        # Replace these by setting mask values to 0
            np.ones(self.n_continuous_covariate_dims, dtype=np.float32),          # Keep these by setting mask values to 1
            np.ones(self.n_sparse_continuous_covariate_dims, dtype=np.float32)    # Keep these by setting mask values to 1
        ])  # Implementation details: covariates_used = original_covariates * mask + replacement * (1 - mask), so categoricals mask values are zeros for replacing the originals
        self.initialized_features = True
        print(f"Encoding covariates with: {self.obs_encoding_dict, self.sparse_continuous_covariates_dict}")

        return self.initialized_features

    def prepare_data(self):
        if not self.initialized_features:
            print("Warning: Features not initialized. Calling self.initialize_features()")
            self.initialize_features()

        print(f'Preparing training data using {len(self.adata)} adatas: {self.adata}', flush=True)
        self.train_adataset = self._build_training_adataset(self.memory_mode)
        self.prepared_data = True

        return self.train_adataset

    def prepare_model(self, **model_kwargs):
        """
        Build the model and put it on the Composer's device (self.device), so inference works before training too.
        Keyword arguments override the Composer's own model kwargs (self.model_kwargs), so prepare_model() with no
        arguments builds the model the Composer was configured with.
        """
        if not self.prepared_data:
            print("Warning: Data not prepared. Calling self.prepare_data()")
            self.prepare_data()

        # Start from the Composer's model kwargs, then update them with covariate dimensions
        model_kwargs = {**self.model_kwargs, **model_kwargs}
        model_kwargs['n_categorical_covariate_dims'] = self.n_categorical_covariate_dims
        model_kwargs['n_total_covariate_dims'] = self.n_total_covariate_dims
        model_kwargs['input_size'] = self.input_size

        # Initialize model
        match self.distribution:
            case 'nb' | 'zinb':
                pass
            case _:
                raise NotImplementedError('ERROR: Only NB and ZINB distributions are currently supported.')
        # Initialize model
        match self.parameterization:
            case 'ksi-psi':
                self.model = Etude(**model_kwargs)
            case 'mu-theta':
                self.model = EtudeMuTheta(**model_kwargs)
            case _:  # Checked at Composer creation; reachable for a Composer pickled before that check
                self._canonical_parameterization(self.parameterization)
        # The weights are built on the CPU above and moved afterwards, so they do not depend on the device
        self._move_model_to_device(self.device)
        print(
            f'Preparing model with input size: {self.input_size}, distribution: {self.distribution}, '
            f'categorical_covariate_keys: {self.categorical_covariate_keys}, continuous_covariate_keys: {self.continuous_covariate_keys}, '
            f'adversarial: {model_kwargs["adversarial"]}'
        )
        self.prepared_model = True
        self.trained_model = False  # The new model has random weights until train() or load_model()

        return self.model

    def deepcopy_model(self):
        assert self.prepared_model

        return copy.deepcopy(self.model)

    def train(self, early_stopping_metric='ref'):
        # Check the metric before any training, checkpoint or NVTX range
        early_stopping_metric = self._canonicalize_early_stopping_metric(early_stopping_metric)

        nvtx.range_push(f"Train model")

        optimizer, n_batches, n_samples = self._initialize_training()
        compiled_train_step = self._compile_train_step()
        prev_loss = torch.tensor(torch.inf, dtype=torch.float32, device=self.device)
        n_epochs_no_improvement = torch.zeros((), dtype=torch.float32, device=self.device)

        nvtx.range_push(f"Train epoch {0}")
        epoch_losses = {k: torch.zeros((), dtype=torch.float32, device=self.device) for k in self.LOSS_KEYS}
        kld_weight_ = torch.zeros((), dtype=torch.float32, device=self.device)
        adv_weight_ = torch.zeros((), dtype=torch.float32, device=self.device)
        best_epoch, best_loss, best_model_weights = 0, torch.tensor(torch.inf, dtype=torch.float32, device=self.device), None
        for epoch_idx in range(self.max_epochs):
            nvtx.range_push("Reset initial epoch losses")
            for _ in epoch_losses.values():
                _.zero_()
            nvtx.range_pop()  # "Reset initial epoch losses"

            nvtx.range_push(f"Train mini-batch {0}")
            nvtx.range_push(f"Load first mini-batch")
            for batch_idx, batch in tqdm(
                enumerate(self.train_adata_loader),
                desc=f"Epoch {epoch_idx}/{self.max_epochs}: ", unit="batch",
                total=len(self.train_adata_loader)
            ):
                batch = batch.to(device=self.device, non_blocking=True) # For non-GPU memory modes
                nvtx.range_pop()  # "Load first mini-batch"; "Load next mini-batch"

                # Train one epoch
                kld_weight_.fill_(self._get_warmup(epoch_idx, batch_idx, n_batches, self.min_kld_weight, self.max_kld_weight, self.n_annealing_epochs))
                adv_weight_.fill_(self._get_warmup(epoch_idx, batch_idx, n_batches, self.min_adv_weight, self.max_adv_weight, self.n_annealing_epochs))
                losses_dict = compiled_train_step(self.model, optimizer, batch, kld_weight=kld_weight_, adv_weight=adv_weight_)

                # Track loss after .backward() is called for graph to be already detached
                nvtx.range_push(f"Save mini-batch {batch_idx} losses")
                for _ in losses_dict:
                    epoch_losses[_] += losses_dict[_]
                nvtx.range_pop()  # "Save mini-batch {batch_idx} losses"

                # Batch range pop/push
                nvtx.range_pop()  # "Train mini-batch {0}"; "Train mini-batch {batch_idx + 1}"
                nvtx.range_push(f"Train mini-batch {batch_idx + 1}")
                nvtx.range_push(f"Load next mini-batch")
            # Scale every key epoch_losses was built from, not only the keys the last mini-batch returned
            for k in self.LOSS_KEYS:
                epoch_losses[k] /= n_samples
            nvtx.range_pop()  # Clear "Load next mini-batch"
            nvtx.range_pop()  # Clear "Train mini-batch {batch_idx + 1}"

            self._print_epoch_losses(epoch_losses, kld_weight_, adv_weight_)
            self._save_model_checkpoint(epoch_idx)
            if epoch_losses[early_stopping_metric] < best_loss:
                best_epoch = epoch_idx
                best_loss.fill_(epoch_losses[early_stopping_metric])
                best_model_weights = copy.deepcopy(self.model.state_dict())
            nvtx.range_pop()  # "Train epoch {0}"; "Train epoch {epoch_idx + 1}"

            if self._early_stopping(n_epochs_no_improvement, epoch_losses[early_stopping_metric], prev_loss):
                break
            nvtx.range_push(f"Train epoch {epoch_idx + 1}")
        nvtx.range_pop()  # Extra "Train epoch {epoch_idx + 1}"
        self._save_trained_model(best_model_weights, best_epoch=best_epoch)

        nvtx.range_pop()  # "Train model"

        return self.model

    def _initialize_training(self):
        if not self.prepared_model:
            print("Warning: Model not initialized. Calling self.prepare_model(**self.model_kwargs)")
            self.prepare_model(**self.model_kwargs)

        nvtx.range_push("Initialize training")

        # Create output directories
        self.checkpoint_path = f'{self.outdir}/checkpoints'
        os.makedirs(f'{self.checkpoint_path}', exist_ok=True)

        # Prepare dataloaders
        nvtx.range_push("Prepare Dataloaders")
        seed_worker, dataloader_generator = self.set_determinism()
        # num_workers is 0 in GPU memory modes: _check_num_workers refuses anything else at Composer creation
        self.train_adata_loader = DataLoader(
            self.train_adataset, batch_size=None, num_workers=self.num_workers,
            sampler=self._get_sampler(self.train_adataset, batch_size=self.batch_size, samples_per_class=self.samples_per_class, shuffle=self.shuffle, drop_last=self.drop_last, memory_mode=self.memory_mode),
            worker_init_fn=seed_worker,
            generator=dataloader_generator,
            persistent_workers = self.num_workers > 0,
            pin_memory=(self.memory_mode not in ('GPU', 'SparseGPU')),  # speeds up host to device copy
        )
        # Batches and cells per epoch as the sampler draws them; with stratified sampling, an epoch has fewer than all cells
        sampler = self.train_adata_loader.sampler
        n_batches = len(sampler)
        n_samples = n_batches * self.batch_size if self.drop_last else sampler.total_samples
        nvtx.range_pop()  # "Prepare Dataloaders"

        nvtx.range_push("Prepare optimizer")
        print(f'Training started using device={self.device} and memory_mode={self.memory_mode} with up to {self.max_epochs} epochs and random seed {self.random_seed} for run version: {self.run_name}', flush=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model = self.model.to(device=self.device)
        self.model.train()  # Set to training mode, as opposed to .eval()
        nvtx.range_pop()  # "Prepare optimizer"

        nvtx.range_push("Save model weights")
        if self.save_initial_weights or self.checkpoint_every_n_epochs is not None:
            torch.save(self.model.state_dict(), f'{self.checkpoint_path}/model_epoch=-1.pt')
            print(f'Model saved at {self.checkpoint_path}/model_epoch=-1.pt', flush=True)
        nvtx.range_pop()  # "Save model weights"

        nvtx.range_pop()  # "Initialize training"

        return optimizer, n_batches, n_samples

    def _compile_train_step(self, training_metric='total'):
        nvtx.range_push("Compile train step")

        def train_step(model, optimizer, batch, kld_weight, adv_weight, training_metric=training_metric):
            # Forward pass
            losses_dict = model.training_step(batch, kld_weight, adv_weight)
            total_ = losses_dict[training_metric]

            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # Should already be default but make explicit
            total_.backward()
            optimizer.step()

            metrics = {k: v.detach() for k, v in losses_dict.items()}  # Detaches from compute graph
            return metrics

        if torch.cuda.is_available() and self.compile_model:
            compiled_train_step = torch.compile(train_step, mode='max-autotune')  # , fullgraph=True)
            print('Model compiling for up to 10x faster training', flush=True)
        else:
            compiled_train_step = train_step
            print('CUDA not available or compilation turned off', flush=True)

        nvtx.range_pop()  # "Compile train step"

        return compiled_train_step

    def _print_epoch_losses(self, epoch_losses, kld_weight, adv_weight, flush=False):
        nvtx.range_push("Print epoch losses")
        msg = 'Epoch '
        msg += ', '.join(f"{k.capitalize()}: {epoch_losses[k]:.3f}" for k in self.LOSS_KEYS)
        msg += f", KLD weight: {kld_weight:.6f}"
        msg += f", ADV weight: {adv_weight:.6f}"
        print(msg, flush=flush)
        nvtx.range_pop()  # "Print epoch losses"

        return msg

    def _save_model_checkpoint(self, epoch_idx):
        nvtx.range_push("Possibly saving model checkpoint")
        if self.checkpoint_every_n_epochs is not None and (epoch_idx + 1) % self.checkpoint_every_n_epochs == 0:
            torch.save(self.model.state_dict(), f'{self.checkpoint_path}/model_epoch={epoch_idx}.pt')
        nvtx.range_pop()  # "Possibly saving model checkpoint"

    def _early_stopping(self, n_epochs_no_improvement, curr_loss, prev_loss):
        nvtx.range_push("Checking for early stopping")
        if not self.early_stopping:
            return False

        trigger_early_stopping = False
        curr_delta = prev_loss - curr_loss
        if curr_delta >= self.min_delta:
            print(f'Epoch improvement of {curr_delta:.3f} >= min_delta of {self.min_delta:.3f}')
            prev_loss.fill_(curr_loss)
            n_epochs_no_improvement.zero_()
        else:
            n_epochs_no_improvement.fill_(n_epochs_no_improvement + 1)
            if n_epochs_no_improvement >= self.patience:
                print(f'No improvement in the last {self.patience} epochs. Early stopping')
                trigger_early_stopping = True
        nvtx.range_pop()  # "Checking for early stopping"

        return trigger_early_stopping

    def _save_trained_model(self, best_model_weights, best_epoch=None):
        # No best weights means no epoch had a finite loss (max_epochs < 1 is rejected when the Composer is created); saving
        # None would only fail when it is loaded
        if best_model_weights is None:
            raise RuntimeError(
                "Training has no best model to save: the loss was not finite (NaN or infinite) in any epoch. "
                "Check that counts and covariates are finite, or lower lr if training diverged"
            )
        nvtx.range_push("Save model and var_names")
        self.var_names.to_series().to_csv(
            f'{self.checkpoint_path}/var_names.csv', 
            index=False,
            header=False,
        )
        torch.save(best_model_weights, f'{self.checkpoint_path}/model_checkpoint.pt')
        if best_epoch is not None:
            print(f'Best model at epoch {best_epoch} saved to {self.checkpoint_path}/model_checkpoint.pt', flush=True)

        # Use the saved best weights from here on too, so results in this session match a model reloaded from the checkpoint
        self.model.load_state_dict(best_model_weights)
        self.trained_model = True
        nvtx.range_pop()  # "Save model and var_names"

    # ======================
    # Internal utilities
    # ======================
    def _get_adataset(
        self,
        adata=None,
        memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'] | None] = None,
    ):
        if not self.initialized_features:
            print("Warning: Features not initialized. Calling self.initialize_features()")
            self.initialize_features()

        if memory_mode is None:
            memory_mode = self.memory_mode
        if adata is None:
            return self._get_training_adataset(memory_mode)

        # Check the data with the same checks as training data, for the requested memory mode, before changing any state
        self._check_adata(adata, 'query', memory_mode)
        # The query must have every training gene, checked here so the message is the same in every memory mode
        # (backed's BackedAnnDataset.set_var_subset stays as a fallback for the dataset used on its own)
        self._check_adata_has_genes(adata, 'query', self.var_names)
        # Also for get_latent_representation, which does not use covariates: the user is responsible for valid data
        self._check_categorical_covariates(adata, 'query')
        self._check_continuous_covariates(adata, 'query')
        # Every cell, also those whose category was not seen in training, so one rule covers training and query data
        # (how unseen categories are treated is SCRUM-34's)
        self._check_sparse_continuous_covariates(adata, 'query')

        # Update memory mode
        prev_memory_mode = self.memory_mode
        self._set_adataset_builder(memory_mode)

        # Update data
        # The builder binds the Composer's stratify_column for training; queries override it with None, since only
        # training samples by class, and a query may lack the column or hold NaN in it
        if memory_mode != 'backed':
            if not (len(adata.var_names) == len(self.var_names) and (adata.var_names == self.var_names).all()):
                # adata.X is is copied by _adataset_builder, so passing in a subset view is necessary and sufficient
                print(f"Warning: Subsetting genes to var_names used in training")
                adata = adata[:, self.var_names]
            adataset = self._adataset_builder(adata, stratify_column=None)
        else:
            adataset = self._adataset_builder(adata, stratify_column=None)
            adataset.set_var_subset(self.var_names)

        # Reset memory mode
        self._set_adataset_builder(prev_memory_mode)

        return adataset

    def _get_training_adataset(
        self,
        memory_mode: Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'],
    ):
        """
        Dataset over all training AnnDatas, in the order passed, for memory_mode. The training dataset is reused in the
        training memory mode; another memory mode builds a new dataset on each call rather than storing a second copy.
        """
        self._check_training_adata_kept()
        if self.train_adataset is not None and memory_mode == self.memory_mode:
            print(f"Loading train adataset")
            return self.train_adataset

        # Only the memory mode is checked again: the counts were checked when the Composer was created
        for index, adata in enumerate(self.adata):
            self._check_adata_memory_mode(adata, index, memory_mode)
        return self._build_training_adataset(memory_mode)

    def _check_training_adata_kept(self):
        """Raise if this Composer has no training AnnDatas, so that adata=None has no cells to use."""
        # __init__ always sets a non-empty list of AnnDatas; _state_without_data drops it when a Composer is pickled
        # (Composer.save) or copied (copy.copy, copy.deepcopy), so None means this Composer was rebuilt from one of those
        if self.adata is None:
            raise ValueError(
                "This Composer holds no training AnnData: Composer.save/load and copies of a Composer do not keep it; "
                "pass adata to get_latent_representation / get_counterfactual"
            )

    def _build_training_adataset(
        self,
        memory_mode: Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'],
    ):
        """
        Build one dataset over all training AnnDatas (self.adata), in the order passed, using memory_mode. The dataset
        builder is set for memory_mode only while building, and the Composer's builder is restored afterwards.
        """
        prev_memory_mode = self.memory_mode
        self._set_adataset_builder(memory_mode)

        # One dataset per training AnnData; backed datasets read only the training genes
        train_adatasets = []
        for adata in self.adata:
            adataset = self._adataset_builder(adata)
            if memory_mode == 'backed':
                adataset.set_var_subset(var_subset=self.var_names)
            train_adatasets.append(adataset)

        # Reset memory mode
        self._set_adataset_builder(prev_memory_mode)

        if len(train_adatasets) > 1:
            return ConcatAnnDataset(train_adatasets)
        return train_adatasets[0]

    def _get_num_workers(self, memory_mode: Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'], stacklevel: int) -> int:
        """
        Get the number of DataLoader workers for an inference loader over a dataset in memory_mode.

        In a GPU memory mode this is 0, with a warning if the Composer has num_workers > 0; the Composer's num_workers is
        not changed, so later queries in a CPU memory mode still use the workers.

        :param memory_mode: The memory mode the dataset actually uses (after any CPU fallback).
        :param stacklevel: Passed to warnings.warn, so the warning points at the user's call.
        :return: 0 in a GPU memory mode, otherwise the Composer's num_workers.
        """
        # GPU and SparseGPU datasets hold CUDA tensors, which forked worker processes cannot use
        if memory_mode in ('GPU', 'SparseGPU') and self.num_workers > 0:
            warnings.warn(
                f"memory_mode='{memory_mode}' holds the data on the GPU, which worker processes cannot use: using "
                f"num_workers=0 for this call instead of num_workers={self.num_workers}",
                UserWarning, stacklevel=stacklevel,
            )
            return 0
        return self.num_workers

    def _get_sampler(
        self,
        adataset,
        batch_size: int = 128,
        samples_per_class = None,
        shuffle: bool = False,
        drop_last: bool = False,
        memory_mode: Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'] = None,
    ):
        if memory_mode is None:
            memory_mode = self.memory_mode
        return TensorBatchSampler(
            adataset,
            batch_size=batch_size,
            device='cuda' if memory_mode in ("GPU", "SparseGPU") and torch.cuda.is_available() else 'cpu',
            shuffle=shuffle,
            drop_last=drop_last,
            samples_per_class=samples_per_class,
        )

    def _get_warmup(
        self,
        epoch, batch_idx=0, n_batches=1,
        min_weight=0, max_weight=1.0, n_annealing_epochs=400,
    ):
        # Without annealing epochs, use max_weight from the start
        if n_annealing_epochs == 0:
            return max_weight
        training_progress = epoch + (batch_idx / n_batches)

        # Use a linear beta-annealing schedule, holding max_weight once annealing is done (training may run longer)
        annealed_fraction = min(training_progress / n_annealing_epochs, 1.0)
        return min_weight + annealed_fraction * (max_weight - min_weight)

    def _set_adataset_builder(
        self,
        memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'] | None] = None,
    ):
        """
        Parameters
        ----------
        memory_mode : Union[Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'] | None], optional
            If None, uses self.memory mode (by default)

        Raises
        ------
        NotImplementedError
            Only supports GPU, SparseGPU, 'SparseCPU', CPU, and backed memory modes
        """

        if memory_mode is None:
            memory_mode = self.memory_mode

        adataset_kwargs = dict(
            memory_mode=memory_mode,
            unlabeled=self.unlabeled,  # The datasets otherwise default to 'Unknown'
            categorical_covariate_keys=self.categorical_covariate_keys,
            continuous_covariate_keys=self.continuous_covariate_keys,
            obs_encoding_dict=self.obs_encoding_dict,
            obs_decoding_dict=self.obs_decoding_dict,
            stratify_column=self.stratify_column,
        )

        match memory_mode:
            case 'GPU' | 'CPU':
                self._adataset_builder = partial(AnnDataset, **adataset_kwargs)
            case 'SparseGPU':
                self._adataset_builder = partial(SparseGPUAnnDataset, **adataset_kwargs)
            case 'SparseCPU':
                adataset_kwargs['sparse_continuous_covariate_keys'] = self.sparse_continuous_covariate_keys
                adataset_kwargs['sparse_continuous_covariates_dict'] = self.sparse_continuous_covariates_dict
                self._adataset_builder = partial(SparseCPUAnnDataset, **adataset_kwargs)
            case 'backed':
                self._adataset_builder = partial(BackedAnnDataset, **adataset_kwargs)
            case _:
                raise NotImplementedError('Only GPU, SparseGPU, SparseCPU, CPU, and backed modes are supported')
