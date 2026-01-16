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
import os
import pickle
import random
from typing import Iterable, Literal, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.model_selection import train_test_split
from torch.cuda import nvtx
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

from piano.utils.data import AnnDataset, SparseGPUAnnDataset, BackedAnnDataset, GPUBatchSampler, streaming_hvg_indices
from piano.models.base_models import Etude, PaddedEtude, PaddedZinbEtude, ZinbEtude, scVI


class Composer():
    def __init__(
            self, 
            
            # Training data
            adata,

            # Composer arguments
            memory_mode: Literal['GPU', 'SparseGPU', 'CPU', 'backed'] = 'GPU',
            integration_mode: Literal['PIANO', 'scVI'] = 'PIANO',
            compile_model: bool = True,
            use_padding: bool = False,
            distribution: Literal['nb', 'zinb'] = 'nb',
            cross_validation: bool = False,
            categorical_covariate_keys=None,
            continuous_covariate_keys=None,
            unlabeled: str = 'Unknown',

            # Gene selection
            flavor: str = 'seurat_v3',
            n_top_genes: int = 4096,
            hvg_batch_key=None,
            geneset_path=None,

            # Prepare AnnDatasets
            validation_split: float = 0.0,

            # Model kwargs
            input_size: int = 4096,  # Must be Python int
            n_hidden: int = 256,
            n_layers: int = 3,
            latent_size: int = 32,
            cov_size: int = 0,
            dropout_rate: float = 0.1,
            batchnorm_eps: float = 1e-5,       # Torch default is 1e-5
            batchnorm_momentum: float = 1e-1,  # Torch default is 1e-1
            epsilon: float = 1e-5,             # Torch default is 1e-5

            # Training
            max_epochs: int = 200,
            batch_size: int = 128,
            min_weight: float = 0.00,
            max_weight: float = 1.00,
            cyclic_annealing_m: int = 400,
            anneal_batches: bool = True,
            lr: float = 2e-4,
            weight_decay: float = 0.00,
            save_initial_weights: bool = False,
            checkpoint_every_n_epochs = None,
            early_stopping: bool = True,
            min_delta: float = 1.00,
            patience: int = 5,
            shuffle: bool = True,
            drop_last: bool = True,
            num_workers: int = 0,

            # Reproducibility
            deterministic: bool = True,
            random_seed: int = 0,

            # Output
            run_name: str = 'piano_integration',
            outdir: str = './results/',
        ):

        # Initialize pipeline flags
        self.initialized_features = False
        self.prepared_data = False
        self.prepared_model = False
        self.trained_model = False
        self.cross_validation = cross_validation

        # Save input arguments
        self.adata = adata
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.memory_mode = memory_mode
        if not torch.cuda.is_available() and (self.memory_mode in ('GPU', 'SparseGPU')):
            print("Warning: GPU not available. Setting memory_mode to CPU and device to cpu")
            self.memory_mode = 'CPU'
        self.integration_mode = integration_mode
        assert self.integration_mode in ('PIANO', 'scVI'), 'ERROR: Only PIANO and scVI integrations are currently supported'
        self.compile_model = compile_model
        self.use_padding = use_padding
        self.distribution = distribution.lower()
        if self.distribution not in ('nb', 'zinb'):
            raise NotImplementedError('ERROR: Only NB and ZINB distributions are currently supported')
        self.var_names = None
        self.categorical_covariate_keys = categorical_covariate_keys
        self.continuous_covariate_keys = continuous_covariate_keys
        self.unlabeled = unlabeled
        self.obs_columns_to_keep = self.categorical_covariate_keys + self.continuous_covariate_keys
        self.obs_columns_to_encode = self.categorical_covariate_keys

        # Gene selection
        self.flavor = flavor
        self.n_top_genes = n_top_genes
        self.hvg_batch_key = hvg_batch_key
        self.geneset_path = geneset_path

        # Prepare AnnDatasets
        self.validation_split = validation_split

        # Save model kwargs
        self.model_kwargs = {
            "input_size": input_size,
            "n_hidden": n_hidden,
            "n_layers": n_layers,
            "latent_size": latent_size,
            "cov_size": cov_size,
            "dropout_rate": dropout_rate,
            "batchnorm_eps": batchnorm_eps,
            "batchnorm_momentum": batchnorm_momentum,
            "epsilon": epsilon,
        }

        # Uninitialized encodings
        self.obs_encoding_dict = {}
        self.obs_decoding_dict = {}
        self.obs_zscoring_dict = {}
        self.train_adataset = None
        self.valid_adataset = None
        self.counterfactual_covariates = None

        # Uninitialized model
        self.model = None
        self.train_adata_loader = None
        self.valid_adata_loader = None
        self.checkpoint_path = None

        # Training
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.cyclic_annealing_m = cyclic_annealing_m
        self.anneal_batches = anneal_batches
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_initial_weights = save_initial_weights
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.early_stopping = early_stopping
        self.min_delta = min_delta
        self.patience = patience
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

        # Set seed for reproducibility
        self.deterministic = deterministic
        self.random_seed = random_seed
        if self.deterministic:
            # Set seed
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

            # Use deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

        # Save output
        self.run_name = run_name
        self.outdir = outdir

    def __getstate__(self):
        # Get the object's state (default)
        state = self.__dict__.copy()

        # Remove Anndatas, Datasets, and DataLoaders from state to avoid large pickles
        state['adata'] = None
        state['train_adataset'] = None
        state['valid_adataset'] = None
        state['train_adata_loader'] = None
        state['valid_adata_loader'] = None

        return state

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
        
    def load_model(self, model_checkpoint_path):
        self.model.load_state_dict(
            torch.load(
                model_checkpoint_path,
                weights_only=True
            )
        )

        return self.model

    def initialize_features(self):
        if self.geneset_path is None:
            print(f'Preparing data with parameters: {self.flavor, self.n_top_genes, self.hvg_batch_key}')
        else:
            print(f'Preparing data with gene set: {self.geneset_path}')

        if self.memory_mode == 'backed':
            assert isinstance(self.adata, str), "Only str paths are allowed for backed mode"
            print('initialize_features: backed mode, loading only obs/var metadata', flush=True)

            # Load metadata
            adata_backed = sc.read_h5ad(self.adata, backed='r')
            obs = adata_backed.obs[self.obs_columns_to_keep].copy()
            var_names = adata_backed.var_names.copy()

            # Gene‐set vs HVG subsetting for var_names
            if self.geneset_path is None:
                if self.n_top_genes > 0:
                    var_names = var_names[streaming_hvg_indices(adata_backed, self.n_top_genes)]
            else:
                var_names = np.intersect1d(var_names, pd.read_csv(self.geneset_path, index_col=0).values.ravel())
            self.var_names = var_names
            adata_backed.file.close()

            # Categorical encodings
            for col in self.obs_columns_to_encode:
                values = obs[col].astype(str)
                codes, uniques = pd.factorize(values)
                self.obs_encoding_dict[col] = {u: i for i, u in enumerate(uniques)}
                self.obs_decoding_dict[col] = {i: u for i, u in enumerate(uniques)}

            # Continuous z-scoring
            for col in self.continuous_covariate_keys:
                arr = obs[col].values.astype(np.float32)
                mu, sigma = arr.mean(), arr.std() + arr.mean() * 1e-5
                self.obs_zscoring_dict[col] = (mu, sigma)
            self.initialized_features = True

            return

        # Encode things
        counterfactual_covariates = []
        for obs_column in self.obs_columns_to_encode:
            # Add encoding dict for each obs column
            self.adata.obs[obs_column] = [str(_) for _ in self.adata.obs[obs_column]]
            sorted_labels = sorted(set(self.adata.obs[obs_column]) - set([self.unlabeled]))
            integer_encodings, original_labels = pd.factorize(np.array(sorted_labels))
            self.obs_encoding_dict[obs_column] = {self.unlabeled: -1} \
                | { _[0]:_[1] for _ in zip(original_labels, integer_encodings)}
            self.obs_decoding_dict[obs_column] = {-1: self.unlabeled} \
                | { _[1]:_[0] for _ in zip(original_labels, integer_encodings)}

            # Append [1 / k, ...] for each categorical covariate
            counterfactual_covariates += [1 / len(integer_encodings)] * len(integer_encodings)

        for continuous_covariate_key in self.continuous_covariate_keys:
            data = self.adata.obs[continuous_covariate_key].values.astype(np.float32)
            mean = np.mean(data)
            std = np.std(data) + mean * 1e-5  # Smoothing in case of low variance
            self.obs_zscoring_dict[continuous_covariate_key] = (mean, std)

            # Append 0, since continuous covariates are Z-scored
            counterfactual_covariates.append(0)
        self.counterfactual_covariates = np.array(counterfactual_covariates)

        # Subset to genes of interest
        if self.geneset_path is None:
            if self.hvg_batch_key is not None and self.hvg_batch_key not in self.adata.obs:
                print(f'Unable to find hvg_batch_key {self.hvg_batch_key} in adata.obs for HVG', flush=True)

            if self.n_top_genes > 0:
                sc.pp.highly_variable_genes(
                    self.adata,
                    flavor=self.flavor,
                    n_top_genes=self.n_top_genes,
                    batch_key=self.hvg_batch_key if self.hvg_batch_key in self.adata.obs else None,
                    subset=True,
                    # TODO: Allow specifying layers
                )
        else:
            var_names = np.intersect1d(
                self.adata.var_names, 
                pd.read_csv(self.geneset_path, index_col=0).values.ravel()
            )
            self.adata = self.adata[:, var_names].copy()
        self.var_names = self.adata.var_names
        self.initialized_features = True

    def update_features(self):
        raise NotImplementedError('update_features not yet implemented.')

    def prepare_data(self):
        assert self.initialized_features

        # Requires features to be initialized
        assert self.initialized_features
        if not self.initialized_features:
            self._initialize_features()

        # Backed memory mode
        if self.memory_mode == 'backed':
            assert isinstance(self.adata, str), "Only str paths are allowed for backed mode"
            self.train_adataset = BackedAnnDataset(
                h5ad_path=self.adata,
                cat_covs=self.categorical_covariate_keys,
                cont_covs=self.continuous_covariate_keys,
                var_subset=self.var_names,
                obs_encoding_dict=self.obs_encoding_dict,
                obs_decoding_dict=self.obs_decoding_dict,
                obs_zscoring_dict=self.obs_zscoring_dict,
            )
            self.valid_adataset = None
            self.prepared_data = True
            return
        
        # Load data not in backed mode
        if self.validation_split > 0:
            print('Preparing train/validation split for cross_validation', flush=True)
            train_split, valid_split = train_test_split(
                self.adata.obs_names,                 # Indices to split
                test_size=self.validation_split,      # validation_split proportion
                shuffle=True,                         # Shuffle before splitting
                random_state=self.random_seed,        # Set seed for reproducibility
            )
            if self.memory_mode == 'SparseGPU':
                self.train_adataset = SparseGPUAnnDataset(
                    self.adata[train_split],
                    categorical_covariate_keys=self.categorical_covariate_keys,
                    continuous_covariate_keys=self.continuous_covariate_keys,
                    obs_encoding_dict=self.obs_encoding_dict,
                    obs_decoding_dict=self.obs_decoding_dict,
                    obs_zscoring_dict=self.obs_zscoring_dict,
                )
                self.valid_adataset = SparseGPUAnnDataset(
                    self.adata[valid_split],
                    categorical_covariate_keys=self.categorical_covariate_keys,
                    continuous_covariate_keys=self.continuous_covariate_keys,
                    obs_encoding_dict=self.obs_encoding_dict,
                    obs_decoding_dict=self.obs_decoding_dict,
                    obs_zscoring_dict=self.obs_zscoring_dict,
                )
            else:
                if self.memory_mode == 'SparseGPU':
                    self.train_adataset = SparseGPUAnnDataset(
                        self.adata[train_split],
                        categorical_covariate_keys=self.categorical_covariate_keys,
                        continuous_covariate_keys=self.continuous_covariate_keys,
                        obs_encoding_dict=self.obs_encoding_dict,
                        obs_decoding_dict=self.obs_decoding_dict,
                        obs_zscoring_dict=self.obs_zscoring_dict,
                    )
                    self.valid_adataset = SparseGPUAnnDataset(
                        self.adata[valid_split],
                        categorical_covariate_keys=self.categorical_covariate_keys,
                        continuous_covariate_keys=self.continuous_covariate_keys,
                        obs_encoding_dict=self.obs_encoding_dict,
                        obs_decoding_dict=self.obs_decoding_dict,
                        obs_zscoring_dict=self.obs_zscoring_dict,
                    )
                else:
                    self.train_adataset = AnnDataset(
                        self.adata[train_split], memory_mode=self.memory_mode,
                        categorical_covariate_keys=self.categorical_covariate_keys,
                        continuous_covariate_keys=self.continuous_covariate_keys,
                        obs_encoding_dict=self.obs_encoding_dict,
                        obs_decoding_dict=self.obs_decoding_dict,
                        obs_zscoring_dict=self.obs_zscoring_dict,
                    )
                    self.valid_adataset = AnnDataset(
                        self.adata[valid_split], memory_mode=self.memory_mode,
                        categorical_covariate_keys=self.categorical_covariate_keys,
                        continuous_covariate_keys=self.continuous_covariate_keys,
                        obs_encoding_dict=self.obs_encoding_dict,
                        obs_decoding_dict=self.obs_decoding_dict,
                        obs_zscoring_dict=self.obs_zscoring_dict,
                    )
        else:
            print('Preparing training data', flush=True)
            if self.memory_mode == 'SparseGPU':
                self.train_adataset = SparseGPUAnnDataset(
                    self.adata,
                    categorical_covariate_keys=self.categorical_covariate_keys,
                    continuous_covariate_keys=self.continuous_covariate_keys,
                    obs_encoding_dict=self.obs_encoding_dict,
                    obs_decoding_dict=self.obs_decoding_dict,
                    obs_zscoring_dict=self.obs_zscoring_dict,
                )
            else:
                self.train_adataset = AnnDataset(
                    self.adata, memory_mode=self.memory_mode,
                    categorical_covariate_keys=self.categorical_covariate_keys,
                    continuous_covariate_keys=self.continuous_covariate_keys,
                    obs_encoding_dict=self.obs_encoding_dict,
                    obs_decoding_dict=self.obs_decoding_dict,
                    obs_zscoring_dict=self.obs_zscoring_dict,
                )
            self.valid_adataset = None
        self.prepared_data = True

    def get_adataset(
        self,
        adata=None,
        memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'backed'] | None] = None,
    ):
        assert self.initialized_features
        if memory_mode is None:
            memory_mode = self.memory_mode

        if adata is not None or memory_mode != self.memory_mode:
            # Create a new AnnDataset using new data or different memory mode
            if memory_mode == 'SparseGPU':
                return SparseGPUAnnDataset(
                    adata[:, self.var_names],
                    categorical_covariate_keys=self.categorical_covariate_keys,
                    continuous_covariate_keys=self.continuous_covariate_keys,
                    obs_encoding_dict=self.obs_encoding_dict,
                    obs_decoding_dict=self.obs_decoding_dict,
                    obs_zscoring_dict=self.obs_zscoring_dict,
                )
            else:
                return AnnDataset(
                    adata[:, self.var_names],
                    memory_mode=memory_mode,
                    categorical_covariate_keys=self.categorical_covariate_keys,
                    continuous_covariate_keys=self.continuous_covariate_keys,
                    obs_encoding_dict=self.obs_encoding_dict,
                    obs_decoding_dict=self.obs_decoding_dict,
                    obs_zscoring_dict=self.obs_zscoring_dict,
                )
        elif self.train_adataset is not None:
            # Use existing AnnDataset
            return self.train_adataset
        else:
            # Only possible by initializing features but not preparing data
            assert self.adata is not None
            if memory_mode == 'SparseGPU':
                return SparseGPUAnnDataset(
                    adata,
                    categorical_covariate_keys=self.categorical_covariate_keys,
                    continuous_covariate_keys=self.continuous_covariate_keys,
                    obs_encoding_dict=self.obs_encoding_dict,
                    obs_decoding_dict=self.obs_decoding_dict,
                    obs_zscoring_dict=self.obs_zscoring_dict,
                )
            else:
                return AnnDataset(
                    adata,
                    memory_mode=memory_mode,
                    categorical_covariate_keys=self.categorical_covariate_keys,
                    continuous_covariate_keys=self.continuous_covariate_keys,
                    obs_encoding_dict=self.obs_encoding_dict,
                    obs_decoding_dict=self.obs_decoding_dict,
                    obs_zscoring_dict=self.obs_zscoring_dict,
                )

    def prepare_model(self, **model_kwargs):
        assert self.prepared_data

        # Compute padding size
        input_size = len(self.var_names)
        if self.use_padding and self.compile_model and input_size % 4 != 0:
            padding_size = 4 - (input_size % 4)
            if 'padding_size' in model_kwargs:
                print(f'Warning: Padding should be determined automatically for torch.compile')
            print(f'Warning: Adding padding {padding_size} for torch.compile')
            model_kwargs['padding_size'] = padding_size
        else:
            padding_size = None

        # Prepare categorical keys
        categorical_covariate_keys = [(_, max(self.obs_encoding_dict[_].values()) + 1)
            for _ in self.categorical_covariate_keys
        ]
        continuous_covariate_keys = self.continuous_covariate_keys
        n_categorical_covariate_dims = int(np.sum([_[1] for _ in categorical_covariate_keys]))
        n_continuous_covariate_dims = len(self.continuous_covariate_keys)
        cov_size = n_categorical_covariate_dims + n_continuous_covariate_dims
        print(
            f'Preparing model with input size: {input_size}, '
            f'distribution: {self.distribution}, '
            f'padding size: {padding_size}, '
            f'categorical_covariate_keys: {categorical_covariate_keys}, '
            f'continuous_covariate_keys: {continuous_covariate_keys}'
        )

        # Override input and covariate_size parameters
        for param_name, param_value in zip(
            ['input_size', 'cov_size'],
            [input_size, cov_size]):
            if param_name in model_kwargs:
                print(f"Warning: {param_name} is overrided by Composer to {param_value}")
            model_kwargs[param_name] = param_value

        # Initialize model
        if self.integration_mode == 'PIANO':
            if not self.use_padding:
                if self.distribution == 'nb':
                    self.model = Etude(
                        **model_kwargs,
                    )
                elif self.distribution == 'zinb':
                    self.model = ZinbEtude(
                        **model_kwargs,
                    )
                else:
                    raise NotImplementedError('ERROR: Only NB and ZINB distributions are currently supported')
            else:
                if self.distribution == 'nb':
                    self.model = PaddedEtude(
                        **model_kwargs,
                    )
                elif self.distribution == 'zinb':
                    self.model = PaddedZinbEtude(
                        **model_kwargs,
                    )
                else:
                    raise NotImplementedError('ERROR: Only NB and ZINB distributions are currently supported')
        elif self.integration_mode == 'scVI':
            if len(self.categorical_covariate_keys) > 0:
                batch_key = self.categorical_covariate_keys[0]
                n_batch_keys = int(max(self.obs_encoding_dict[batch_key].values()) + 1)
                print(f'Using scVI mode with batch key: {batch_key}', flush=True)
            else:
                n_batch_keys = 0
                print('Using scVI mode with no batch key', flush=True)
            self.model = scVI(
                n_batch_keys=n_batch_keys,
                **model_kwargs,
            )
        else:
            raise NotImplementedError('ERROR: Only PIANO and scVI integrations are currently supported')
        self.prepared_model = True

    def deepcopy_model(self):
        assert self.prepared_model

        return copy.deepcopy(self.model)

    def get_warmup(
        self,
        epoch, batch_idx=0, n_batches=1,
        min_weight=0, max_weight=1.0, cyclic_annealing_m=400,
    ):
        training_progress = epoch + (batch_idx / n_batches)
        return min_weight + training_progress / cyclic_annealing_m * (max_weight - min_weight)

    def train_model(self):
        assert self.prepared_model

        # Toggle num_workers based on GPU availability
        if self.memory_mode == 'GPU' and self.num_workers > 0:
            print("Warning: Setting num workers to 0 for GPU memory mode")
            self.num_workers = 0
        nvtx.range_push("Prepare to train model")
        print(f'Training model with up to {self.max_epochs} epochs and random seed: {self.random_seed}', flush=True)

        # Set seed for reproducibility
        if self.deterministic:
            # Set seed
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)

            # Use deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

            # Deterministic workers
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            dataloader_generator = torch.Generator()
            dataloader_generator.manual_seed(self.random_seed)

        # Create output directories
        self.checkpoint_path = f'{self.outdir}/checkpoints'
        os.makedirs(f'{self.checkpoint_path}', exist_ok=True)
        nvtx.range_pop()

        # Prepare dataloaders
        nvtx.range_push("Prepare Dataloaders")

        # This branch for backed might not be needed; see args
        if self.memory_mode == 'backed':
            self.train_adata_loader = DataLoader(
                self.train_adataset,
                batch_size=None,
                num_workers=self.num_workers,
                sampler=BatchSampler(
                    (
                        RandomSampler(self.train_adataset) if self.shuffle else
                        SequentialSampler(self.train_adataset)
                    ),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                ),
                worker_init_fn=seed_worker if self.deterministic else None,
                generator=dataloader_generator if self.deterministic else None,
                persistent_workers = self.num_workers > 0,
                pin_memory=(self.memory_mode not in ('GPU', 'SparseGPU')),  # speeds up host to device copy
            )
        elif self.cross_validation:
            self.train_adata_loader = DataLoader(
                self.train_adataset,
                batch_size=None,
                num_workers=self.num_workers,
                sampler=BatchSampler(
                    (
                        RandomSampler(self.train_adataset) if self.shuffle else
                        SequentialSampler(self.train_adataset)
                    ),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                ),
                worker_init_fn=seed_worker if self.deterministic else None,
                generator=dataloader_generator if self.deterministic else None,
                persistent_workers = self.num_workers > 0,
                pin_memory=(self.memory_mode not in ('GPU', 'SparseGPU')),  # speeds up host to device copy
            )
            self.valid_adata_loader = DataLoader(
                self.valid_adataset,
                batch_size=None,
                num_workers=self.num_workers,
                sampler=BatchSampler(
                    (
                        RandomSampler(self.valid_adataset) if self.shuffle else
                        SequentialSampler(self.valid_adataset)
                    ),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                ),
                worker_init_fn=seed_worker if self.deterministic else None,
                generator=dataloader_generator if self.deterministic else None,
                persistent_workers = self.num_workers > 0,
                pin_memory=(self.memory_mode not in ('GPU', 'SparseGPU')),  # speeds up host to device copy
            )
        else:
            self.train_adata_loader = DataLoader(
                self.train_adataset,
                batch_size=None,
                num_workers=self.num_workers,
                sampler=(
                    GPUBatchSampler(
                        self.train_adataset,
                        batch_size=self.batch_size,
                        drop_last=self.drop_last,
                    ) if self.memory_mode == 'GPU' and torch.cuda.is_available() else
                    BatchSampler(
                        (
                            RandomSampler(self.train_adataset) if self.shuffle else
                            SequentialSampler(self.train_adataset)
                        ),
                        batch_size=self.batch_size,
                        drop_last=self.drop_last,
                    )
                ),
                worker_init_fn=seed_worker if self.deterministic else None,
                generator=dataloader_generator if self.deterministic else None,
                persistent_workers = self.num_workers > 0,
                pin_memory=(self.memory_mode not in ('GPU', 'SparseGPU')),  # speeds up host to device copy
            )
        nvtx.range_pop()

        # Save model weights
        if self.save_initial_weights or self.checkpoint_every_n_epochs is not None:
            torch.save(self.model.state_dict(), f'{self.checkpoint_path}/model_epoch=-1.pt')
            print(f'Model saved at {self.checkpoint_path}/model_epoch=-1.pt', flush=True)

        nvtx.range_push("Start fitting model")
        print(
            f'Training started for {self.run_name} '
            f'using device={self.device} and '
            f'memory_mode={self.memory_mode}'
        , flush=True)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.model = self.model.to(device=self.device)
        self.model.train()

        total_cells = len(self.train_adataset)
        if self.drop_last:
            n_batches = total_cells // self.batch_size
            n_samples = n_batches * self.batch_size
        else:
            n_batches = (total_cells + self.batch_size - 1) // self.batch_size
            n_samples = len(self.train_adataset)

        # Compile model for faster training
        nvtx.range_push("torch.compile")
        def train_step(model, optimizer, batch, kld_weight, adv_lambda):
            # Forward pass
            elb_, nll_, kld_, adv_ = model.training_step(batch, kld_weight, adv_lambda)

            # Backward pass
            optimizer.zero_grad()
            elb_.backward()
            optimizer.step()

            return elb_, nll_, kld_, adv_
        if torch.cuda.is_available() and self.compile_model:
            compiled_train_step = torch.compile(train_step, mode="max-autotune")  # , fullgraph=True)
            print('Model compiling for up to 10x faster training', flush=True)
        else:
            compiled_train_step = train_step
            print('CUDA not available or compilation turned off', flush=True)
        nvtx.range_pop()

        # Early stopping initiliaiztion
        if self.early_stopping:
            prev_loss = torch.tensor(torch.inf, dtype=torch.float32, device='cuda')
            n_epochs_no_improvement = 0

        nvtx.range_push(f"Epoch {0}")
        epoch_elbo = torch.tensor(0, dtype=torch.float32, device='cuda')
        epoch_nll = torch.tensor(0, dtype=torch.float32, device='cuda')
        epoch_kld = torch.tensor(0, dtype=torch.float32, device='cuda')
        epoch_adv = torch.tensor(0, dtype=torch.float32, device='cuda')
        kld_weight = torch.tensor(0, dtype=torch.float32, device='cuda')
        best_epoch = 0
        best_elbo = torch.tensor(torch.inf, dtype=torch.float32, device='cuda')
        best_model_weights = None
        for epoch_idx in range(self.max_epochs):
            nvtx.range_push("Pre-training time")
            epoch_elbo.fill_(0)
            epoch_nll.fill_(0)
            epoch_kld.fill_(0)
            epoch_adv.fill_(0)
            nvtx.range_pop()

            nvtx.range_push(f"Batch {0}")
            nvtx.range_push(f"Get data")
            for batch_idx, batch in tqdm(
                enumerate(self.train_adata_loader),
                desc=f"Epoch {epoch_idx}/{self.max_epochs}: ", unit="batch",
                total=len(self.train_adata_loader)
            ):
                batch = batch.to(device=self.device, non_blocking=True) # For non-GPU memory modes
                nvtx.range_pop()

                # Train one epoch
                if self.anneal_batches:
                    kld_weight.fill_(
                        self.get_warmup(
                            epoch_idx, batch_idx, n_batches,
                            min_weight=self.min_weight,
                            max_weight=self.max_weight,
                            cyclic_annealing_m=self.cyclic_annealing_m,
                        )
                    )
                else:
                    kld_weight.fill_(
                        self.get_warmup(
                            epoch_idx,
                            min_weight=self.min_weight,
                            max_weight=self.max_weight,
                            cyclic_annealing_m=self.cyclic_annealing_m,
                        )
                    )
                elb_, nll_, kld_, adv_ = compiled_train_step(self.model, optimizer, batch, kld_weight, adv_lambda=kld_weight)

                # Track loss after .backward() is called for graph to be already detached
                nvtx.range_push("Save loss step")
                epoch_elbo += elb_
                epoch_nll += nll_
                epoch_kld += kld_
                epoch_adv += adv_
                nvtx.range_pop()

                # Batch range pop/push
                nvtx.range_pop()
                nvtx.range_push(f"Batch {batch_idx + 1}")
                nvtx.range_push(f"Get data")
            epoch_elbo /= n_samples
            epoch_nll /= n_samples
            epoch_kld /= n_samples
            epoch_adv /= n_samples
            nvtx.range_pop()
            nvtx.range_pop()

            nvtx.range_push("Print loss epoch")
            print(
                f"Epoch ELBO: {(epoch_elbo):.3f}, "
                f"NLL: {(epoch_nll):.3f}, "
                f"KLD: {(epoch_kld):.3f}, "
                f"ADV: {(epoch_adv):.3f}, "
                f"KLD weight: {kld_weight:.6f}"
            )
            nvtx.range_pop()

            # Model checkpointing
            nvtx.range_push("Saving model checkpoint")
            if (
                self.checkpoint_every_n_epochs is not None
                and (epoch_idx + 1) % self.checkpoint_every_n_epochs == 0
            ):
                torch.save(
                    self.model.state_dict(), 
                    f'{self.checkpoint_path}/model_epoch={epoch_idx}.pt',
                )
            nvtx.range_pop()

            nvtx.range_pop()  # nvtx.range_push(f"Epoch {epoch_idx}")
            nvtx.range_push(f"Epoch {epoch_idx + 1}")

            # Update best epoch
            if epoch_elbo < best_elbo:
                best_epoch = epoch_idx
                best_elbo.fill_(epoch_elbo)
                best_model_weights = copy.deepcopy(self.model.state_dict())

            # Early stopping
            if self.early_stopping:
                curr_delta = prev_loss - epoch_elbo
                if curr_delta >= self.min_delta:
                    print(f'Epoch improvement of {curr_delta:.3f} >= min_delta of {self.min_delta:.3f}')
                    prev_loss.fill_(epoch_elbo)
                    n_epochs_no_improvement = 0
                else:
                    n_epochs_no_improvement += 1
                    if n_epochs_no_improvement >= self.patience:
                        print(f'No improvement in the last {self.patience} epochs. Early stopping')
                        break
        print(
            f'Training completed for {self.run_name} '
            f'using device={self.device} and '
            f'memory_mode={self.memory_mode}'
        , flush=True)
        nvtx.range_pop()

        # Save model gene names and parameter weights
        nvtx.range_push("Save var_names")
        self.var_names.to_series().to_csv(
            f'{self.checkpoint_path}/var_names.csv', 
            index=False,
            header=False,
        )
        torch.save(best_model_weights, f'{self.checkpoint_path}/model_checkpoint.pt')
        print(f'Best model at epoch {best_epoch} saved to {self.checkpoint_path}/model_checkpoint.pt', flush=True)
        nvtx.range_pop()
        self.trained_model = True

    def train(self):
        # Alias for self.train_model
        self.train_model()

    def fit(self):
        # Alias for self.train_model
        self.train_model()

    def get_latent_representation(
        self,
        adata=None,
        memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'backed'] | None] = None,
        mc_samples=0,
    ):
        adataset = self.get_adataset(adata, memory_mode)
        adata_loader = DataLoader(
            adataset,
            batch_size=None,
            num_workers=self.num_workers,
            sampler=BatchSampler(
                SequentialSampler(adataset),
                batch_size=4096,
                drop_last=False,
            )
        )
        latent_space = self.model.get_latent_representation(
            adata_loader, mc_samples=mc_samples,
        ).cpu().numpy()
        print(f'Retrieving latent space with dims {latent_space.shape}', flush=True)
        
        return latent_space

    def get_counterfactual(
        self,
        adata=None,
        covariates: Union[Literal['marginal'] | Iterable[float] | None] = 'marginal',
        memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'backed'] | None] = None,
    ):
        adataset = self.get_adataset(adata, memory_mode)
        adata_loader = DataLoader(
            adataset,
            batch_size=None,
            num_workers=self.num_workers,
            sampler=BatchSampler(
                SequentialSampler(adataset),
                batch_size=4096,
                drop_last=False,
            )
        )
        if covariates == 'marginal':
            covariates = self.counterfactual_covariates
        counterfactuals = self.model.get_counterfactuals(
            adata_loader, covariates=covariates,
        ).cpu().numpy()
        print(f'Retrieving counterfactuals with dims {counterfactuals.shape}', flush=True)

        return counterfactuals

    def run_pipeline(self):
        self.initialize_features()
        self.prepare_data()
        self.prepare_model(**self.model_kwargs)
        self.train_model()
