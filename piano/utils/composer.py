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
import os
import pickle
import random
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
from piano.utils.covariates import encode_categorical_covariates, encode_sparse_continuous_covariates
from piano.utils.data import AnnDataset, SparseGPUAnnDataset, SparseCPUAnnDataset, BackedAnnDataset, ConcatAnnDataset, TensorBatchSampler, count_batches, count_cells_per_epoch
from piano.utils.preprocessing import highly_variable_genes


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
        memory_mode: Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'] = 'GPU',
        compile_model: bool = True,
        categorical_covariate_keys=None,
        continuous_covariate_keys=None,
        sparse_continuous_covariate_keys=None,
        unlabeled: str = 'Unknown',

        # Gene selection
        flavor: str = 'seurat_v3',  # Only Seurat V3 is supported (for multiple AnnDatas)
        n_top_genes: int = 4096,
        hvg_batch_key: str = None,
        geneset_path: str = None,

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
        random_seed: int = 0,

        # Output
        run_name: str = 'piano_integration',
        outdir: str = './results/',
    ):
        self._init_params = self._inspect_init_params(locals())
        self._init_pipeline_flags()
        self._init_hardware(self._init_params)
        self._init_data_config(self._init_params)
        self._init_gene_selection(self._init_params)
        self._init_model_config(self._init_params)
        self._disable_adversarial_without_categorical_covariates()
        self._init_training_config(self._init_params)
        self._init_output_config(self._init_params)
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
            if self.memory_mode in ('GPU', 'SparseGPU'):
                print(
                    "Warning: GPU not available. "
                    "Setting memory_mode and device to CPU and not compiling model.")
                self.memory_mode = 'CPU'
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
        self._check_sparse_continuous_covariate_keys(self.memory_mode)
        self.unlabeled = params['unlabeled']

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

        In-memory counts are checked for NaN, infinite and negative values. Backed counts are not checked, as that would
        read the whole file; invalid backed counts fail at the first batch that holds them, with an error from torch.
        """
        if not isinstance(adata, ad.AnnData):
            raise TypeError(
                f"adata {index} is a {type(adata).__name__}, but Composer takes AnnData objects: read the file first, "
                "e.g. ad.read_h5ad(path), or ad.read_h5ad(path, backed='r') for memory_mode='backed'"
            )
        if memory_mode is None:
            memory_mode = self.memory_mode
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
        if not adata.isbacked:
            self._check_count_values(adata.X, index)

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

    def _init_gene_selection(self, params):
        self.flavor = params['flavor']
        self.n_top_genes = params['n_top_genes']
        self.hvg_batch_key = params['hvg_batch_key']
        self.geneset_path = params['geneset_path']
        self.var_names = None

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
        if self.stratify_column is not None and self.samples_per_class is None:
            print(f"Warning: stratify_column '{self.stratify_column}' is set but samples_per_class is None, so training uses all cells without stratification", flush=True)
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

    def _init_output_config(self, params):
        self.run_name = params['run_name']
        self.outdir = params['outdir']

    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove large adata objects to reduce pickle size
        for _ in ['adata', 'train_adataset', 'train_adata_loader']:
            state[_] = None
        # Copy the nested dict too: the shallow copy above shares it with the live Composer
        state['_init_params'] = {**state['_init_params'], 'adata': None}

        # Model weights are already and only saved in model_checkpoint.pt
        # state['model'] = None  # Not removing model to improve user convenience

        return state

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path, model_checkpoint_path=None, device="cpu"):
        """
            This static method loads the Composer from a pickle file and sets the device (Default 'cpu').
            If a model checkpoint path is specified, prepares model from pianist.model_kwargs and loads weights.
            The weights are loaded from checkpoint using load_model, which uses the device specified and sets model to eval.
        """
        with open(path, 'rb') as f:
            pianist = pickle.load(f)
        pianist.device = device

        if model_checkpoint_path is not None:
            pianist.prepare_model(**pianist.model_kwargs)
            pianist.load_model(model_checkpoint_path, device=device)

        return pianist

    def load_model(self, model_checkpoint_path, device="cpu"):
        """
            Loads the model weights from specified checkpoint to specified device.
            This also sets the model to eval mode. To continue training, set .train()
        """
        self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=device, weights_only=True))
        self.device = device
        self.model.device = device
        self.model.to(device)
        self.model.eval()

        return self.model

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

        :param mc_samples: 0 (default) returns each cell's posterior mean, which is deterministic. A positive number returns
            the average of that many random draws from each cell's posterior; mc_samples=1 gives one draw, which differs from
            call to call by about the posterior's standard deviation.
        """
        self._check_mc_samples(mc_samples)
        memory_mode = self.memory_mode if memory_mode is None else self._canonical_memory_mode(memory_mode)

        adataset = self._get_adataset(adata, memory_mode)
        adata_loader = DataLoader(
            adataset, batch_size=None, num_workers=self.num_workers,
            sampler=self._get_sampler(adataset, batch_size=batch_size, samples_per_class=None, shuffle=False, drop_last=False, memory_mode=memory_mode),
        )
        latent_space = self.model.get_latent_representation(
            adata_loader, mc_samples=mc_samples,
        )
        print(f'Retrieving latent space with shape {latent_space.shape}', flush=True)

        return latent_space

    @staticmethod
    def _check_mc_samples(mc_samples):
        """Raise unless mc_samples is 0 (use each cell's posterior mean) or a positive number of posterior draws."""
        if mc_samples < 0:
            raise ValueError(f"mc_samples must be 0 (posterior mean) or a positive number of posterior draws, not {mc_samples}")

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

        :param adata: AnnData to retrieve counterfactual representations
        :param covariates: 
            - marginal (default):
                Compute with categorical covariates averaged per covariate key and continuous covariates using original values
            - dict: {key: value | [value1, value2, ...] | 'marginal'}
                For a given categorical covariate key, there are three options:
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
                If left unspecified, the original value of a covariate is used.
            - None:
                Use the actual covariates in the metadata, which provides a probabilistic reconstruction of the original data.
        :type covariates: Union[Literal['marginal'], dict, None]
        :param batch_size: Number of cells to compute at once
        :type batch_size: int
        :param memory_mode: Memory mode for sampler. Default (None) uses same memory mode as Composer.
        :type memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'SparseCPU', 'backed'] | None]
        :param mc_samples: 0 (default) decodes each cell's posterior mean, so counterfactuals are deterministic. A positive
            number averages the NB means decoded from that many random draws from each cell's posterior (one draw for 1).
        :type mc_samples: int
        :raises ValueError: If covariates is not one of the above, or names a key that is not a covariate of this Composer,
            or if mc_samples is negative.
        """
        self._check_counterfactual_covariates(covariates)  # Before loading data, so typos fail fast
        self._check_mc_samples(mc_samples)
        memory_mode = self.memory_mode if memory_mode is None else self._canonical_memory_mode(memory_mode)
        self._check_sparse_continuous_covariate_keys(memory_mode)  # The decoder needs the covariates the model was trained with

        adataset = self._get_adataset(adata, memory_mode)
        adata_loader = DataLoader(
            adataset, batch_size=None, num_workers=self.num_workers,
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
        """
        encoding_dict = self.obs_encoding_dict[key]
        n_categories = max(encoding_dict.values()) + 1
        if value == 'marginal':
            return np.full(n_categories, 1.0 / n_categories, dtype=np.float32)  # Marginalize over all n possible values

        # Treat one value as a list of one, so both share the encoding below
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            values = list(dict.fromkeys(value))  # remove duplicates, preserve order
        else:
            values = [value]
        k = len(values)
        assert k > 0, f"ERROR: Cannot specify empty list of categorical covariates for covariate: {key}"

        replacement_block = np.zeros(n_categories, dtype=np.float32)
        for v in values:
            if v not in encoding_dict:
                raise ValueError(f"Invalid value '{v}' for covariate '{key}'")
            if encoding_dict[v] < 0:
                continue  # The unlabeled value (encoded -1) is all zeros, as for unlabeled cells in training
            replacement_block[encoding_dict[v]] = 1.0 / k  # Marginalize over k specified values (one-hot if k = 1)

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
            return None, None

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
    def initialize_features(self):
        if self.geneset_path is None:
            print(f'Preparing data with highly_variable_genes flavor = {self.flavor}, {self.n_top_genes} HVGs (-1 = all HVGs if selected before passing into Composer), batch key = {self.hvg_batch_key}')
        else:
            print(f'Preparing data with gene set: {self.geneset_path}')

        if self.memory_mode == 'backed':
            # Select genes with the same rules and gene order as in memory; HVGs read the counts from disk in chunks
            if self.n_top_genes <= 0:  # Use all genes found in first adata
                var_names = self.adata[0].var_names.copy()
            elif self.geneset_path is not None:
                var_names = np.intersect1d(self.adata[0].var_names, pd.read_csv(self.geneset_path, header=None).values.ravel())
            else:
                if self.hvg_batch_key is not None and self.hvg_batch_key not in self.adata[0].obs:
                    print(f'Unable to find hvg_batch_key {self.hvg_batch_key} in adata.obs for HVG', flush=True)
                hvg = highly_variable_genes(
                    self.adata, n_top_genes=self.n_top_genes, inplace=False,
                    batch_key=self.hvg_batch_key if self.hvg_batch_key in self.adata[0].obs else None,
                )
                var_names = hvg.index[hvg['highly_variable'].to_numpy()]

            # Each dataset reads these genes in this order (BackedAnnDataset.set_var_subset), so all must have them
            for index, adata in enumerate(self.adata):
                missing = pd.Index(var_names).difference(adata.var_names)
                if len(missing) > 0:
                    raise ValueError(f"adata {index} is missing {len(missing)} of the selected genes, e.g. {list(missing[:5])}")
        else:
            if self.n_top_genes <= 0:  # Use all genes found in first adata
                var_names = self.adata[0].var_names.copy()
            elif self.geneset_path is not None:
                var_names = np.intersect1d(self.adata[0].var_names, pd.read_csv(self.geneset_path, header=None).values.ravel())
            else:
                if self.hvg_batch_key is not None and self.hvg_batch_key not in self.adata[0].obs:
                    print(f'Unable to find hvg_batch_key {self.hvg_batch_key} in adata.obs for HVG', flush=True)
                highly_variable_genes(
                    self.adata, n_top_genes=self.n_top_genes, subset=True,
                    batch_key=self.hvg_batch_key if self.hvg_batch_key in self.adata[0].obs else None,
                )
                var_names = self.adata[0].var_names.copy()
            for _ in range(len(self.adata)):
                # Subset each adata to feature selected genes
                self.adata[_] = self.adata[_][:, var_names].copy()
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
        self._set_adataset_builder(self.memory_mode)
        train_adatasets = []
        for adata in self.adata:
            adataset = self._adataset_builder(adata)
            if self.memory_mode == 'backed':
                adataset.set_var_subset(var_subset=self.var_names)
            train_adatasets.append(adataset)
        if len(train_adatasets) > 1:
            self.train_adataset = ConcatAnnDataset(train_adatasets)
        else:
            self.train_adataset = train_adatasets[0]
        self.prepared_data = True

        return self.train_adataset

    def prepare_model(self, **model_kwargs):
        """
        Build the model. Keyword arguments override the Composer's own model kwargs (self.model_kwargs), so
        prepare_model() with no arguments builds the model the Composer was configured with.
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
        print(
            f'Preparing model with input size: {self.input_size}, distribution: {self.distribution}, '
            f'categorical_covariate_keys: {self.categorical_covariate_keys}, continuous_covariate_keys: {self.continuous_covariate_keys}, '
            f'adversarial: {model_kwargs["adversarial"]}'
        )
        self.prepared_model = True

        return self.model

    def deepcopy_model(self):
        assert self.prepared_model

        return copy.deepcopy(self.model)

    def train(self, early_stopping_metric='ref'):
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
            for _ in losses_dict:
                epoch_losses[_] /= n_samples
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
        # Toggle num_workers based on GPU availability
        if self.memory_mode == 'GPU' and self.num_workers > 0:
            print("Warning: Setting num workers to 0 for GPU memory mode")
            self.num_workers = 0

        # Create output directories
        self.checkpoint_path = f'{self.outdir}/checkpoints'
        os.makedirs(f'{self.checkpoint_path}', exist_ok=True)

        # Prepare dataloaders
        nvtx.range_push("Prepare Dataloaders")
        seed_worker, dataloader_generator = self.set_determinism()
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

        # Use current train_adataset if no changes to data or memory mode
        same_adataset = (
            self.train_adataset is not None
            and (adata is None or adata is self.adata[0])
            and (memory_mode is None or memory_mode == self.memory_mode)
        )
        if same_adataset:
            print(f"Loading train adataset")
            return self.train_adataset

        # Check the data with the same checks as training data, for the requested memory mode, before changing any state
        if memory_mode is None:
            memory_mode = self.memory_mode
        if adata is None:
            print(f"Warning: No adata passed in. Using first adata in self.adata (list of adata)")
            adata = self.adata[0]
        self._check_adata(adata, 'query', memory_mode)

        # Update memory mode
        prev_memory_mode = self.memory_mode
        self._set_adataset_builder(memory_mode)

        # Update data
        if memory_mode != 'backed':
            if not (len(adata.var_names) == len(self.var_names) and (adata.var_names == self.var_names).all()):
                # adata.X is is copied by _adataset_builder, so passing in a subset view is necessary and sufficient
                print(f"Warning: Subsetting genes to var_names used in training")
                adata = adata[:, self.var_names]
            adataset = self._adataset_builder(adata)
        else:
            adataset = self._adataset_builder(adata)
            adataset.set_var_subset(self.var_names)

        # Reset memory mode
        self._set_adataset_builder(prev_memory_mode)

        return adataset

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
