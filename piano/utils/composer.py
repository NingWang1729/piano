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
from functools import partial
from typing import Iterable, Literal, Union, List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.cuda import nvtx
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

from piano.models.base_models import Etude
from piano.utils.covariates import encode_categorical_covariates, encode_continuous_covariates
from piano.utils.data import AnnDataset, SparseGPUAnnDataset, BackedAnnDataset, ConcatAnnDataset, GPUBatchSampler, streaming_hvg_indices
from piano.utils.preprocessing import highly_variable_genes


class Composer():
    # ======================
    # Construction & I/O
    # ======================
    def __init__(
        self,

        # Training data
        adata: Union[ad.AnnData | List[ad.AnnData]],

        # Composer arguments
        memory_mode: Literal['GPU', 'SparseGPU', 'CPU', 'backed'] = 'GPU',
        compile_model: bool = True,
        categorical_covariate_keys=None,
        continuous_covariate_keys=None,
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

        # Training
        max_epochs: int = 200,
        ## Beta annealing
        batch_size: int = 128,
        min_weight: float = 0.00,
        max_weight: float = 1.00,
        n_annealing_epochs: int = 400,
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
        self.memory_mode = params['memory_mode']
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

    def _init_data_config(self, params):
        if isinstance(params['adata'], (list, tuple)):
            assert len(params['adata']) > 0, 'adata must be an AnnData or non-empty list of AnnDatas'
            self.adata = list(params['adata'])
            if self.memory_mode == 'backed' and len(self.adata) > 1:
                raise NotImplementedError("Backed mode currently supports only a single adata.")
        else:
            self.adata = [params['adata']]
        self.categorical_covariate_keys = params['categorical_covariate_keys'] or []
        self.continuous_covariate_keys = params['continuous_covariate_keys'] or []
        self.obs_columns_to_keep = self.categorical_covariate_keys + self.continuous_covariate_keys
        self.unlabeled = params['unlabeled']

        # Encodings
        self.obs_encoding_dict = {}
        self.obs_decoding_dict = {}
        self.obs_zscoring_dict = {}

        # Objects
        self.counterfactual_categorical_covariates = None
        self.counterfactual_covariates = None
        self.train_adataset = None
        self.train_adata_loader = None

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
        
        # Objects
        self.model = None
        self.checkpoint_path = None

    def _init_training_config(self, params):
        self.max_epochs = params['max_epochs']

        # Beta annealing
        self.batch_size = params['batch_size']
        self.min_weight = params['min_weight']
        self.max_weight = params['max_weight']
        self.n_annealing_epochs = params['n_annealing_epochs']

        # Hyperparameters
        self.lr = params['lr']
        self.weight_decay = params['weight_decay']
        self.shuffle = params['shuffle']
        self.drop_last = params['drop_last']
        self.num_workers = params['num_workers']

        # Early stopping
        self.early_stopping = params['early_stopping']
        self.min_delta = params['min_delta']
        self.patience = params['patience']

        # Checkpoints
        self.save_initial_weights = params['save_initial_weights']
        self.checkpoint_every_n_epochs = params['checkpoint_every_n_epochs']

        # Logging
        self.LOSS_KEYS = ('total', 'elbo', 'nll', 'kld', 'adv')

    def _init_output_config(self, params):
        self.run_name = params['run_name']
        self.outdir = params['outdir']

    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove large data objects to reduce pickle size
        for _ in ['adata', 'train_adataset', 'train_adata_loader']:
            state[_] = None  

        return state

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_model(self, model_checkpoint_path):
        self.model.load_state_dict(torch.load(model_checkpoint_path, weights_only=True))

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
        memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'backed'] | None] = None,
        batch_size: int = 4096,
        mc_samples=0,
    ):
        if memory_mode is None:
            memory_mode = self.memory_mode

        adataset = self._get_adataset(adata, memory_mode)
        adata_loader = DataLoader(
            adataset, batch_size=None, num_workers=self.num_workers,
            sampler=self._get_sampler(adataset, batch_size=batch_size, shuffle=False, drop_last=False, memory_mode=memory_mode),
        )
        latent_space = self.model.get_latent_representation(
            adata_loader, mc_samples=mc_samples,
        )
        print(f'Retrieving latent space with dims {latent_space.shape}', flush=True)

        return latent_space

    def get_counterfactual(
        self,
        adata=None,
        covariates: Union[Literal['marginal'] | Iterable[float] | None] = 'marginal',
        batch_size: int = 4096,
        memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'backed'] | None] = None,
    ):
        if memory_mode is None:
            memory_mode = self.memory_mode

        adataset = self._get_adataset(adata, memory_mode)
        adata_loader = DataLoader(
            adataset, batch_size=None, num_workers=self.num_workers,
            sampler=self._get_sampler(adataset, batch_size=batch_size, shuffle=False, drop_last=False, memory_mode=memory_mode),
        )
        if covariates == 'marginal':
            covariates = self.counterfactual_covariates
        counterfactuals = self.model.get_counterfactuals(
            adata_loader, covariates=covariates,
        )
        print(f'Retrieving counterfactuals with dims {counterfactuals.shape}', flush=True)

        return counterfactuals

    def set_determinism(self, deterministic: bool = None, random_seed: int = None):
        if deterministic is None:
            deterministic = self.deterministic
        else:
            self.deterministic = deterministic

        if not deterministic:
            return None, None

        if random_seed is None:
            random_seed = self.random_seed
        else:
            self.random_seed = random_seed

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

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
        dataloader_generator.manual_seed(random_seed)

        return seed_worker, dataloader_generator

    # ======================
    # Plumbing: Internal API
    # ======================
    def initialize_features(self):
        if self.geneset_path is None:
            print(f'Preparing data with parameters: {self.flavor, self.n_top_genes, self.hvg_batch_key}')
        else:
            print(f'Preparing data with gene set: {self.geneset_path}')

        if self.memory_mode == 'backed':
            assert isinstance(self.adata[0], str), "Only str paths are allowed for backed mode"
            print('initialize_features: backed mode, loading only obs/var metadata', flush=True)
            print('Warning: backed mode only supports using first adata for initialize_features', flush=True)

            # Load adata in backed read mode
            self.adata[0] = sc.read_h5ad(self.adata[0], backed='r')
            print("Warning: Backed mode currently only fully supports 1 adata")
            self.adata = self.adata[:1]  # TODO: Add full support for backed mode

            # Subset to genes of interest
            var_names = self.adata[0].var_names.copy()
            if self.geneset_path is not None:
                var_names = np.intersect1d(var_names, pd.read_csv(self.geneset_path, header=None).values.ravel())
            elif self.n_top_genes > 0:
                var_names = var_names[streaming_hvg_indices(self.adata[0], self.n_top_genes)]
        else:
            if self.geneset_path is not None:
                var_names = np.intersect1d(self.adata[0].var_names, pd.read_csv(self.geneset_path, header=None).values.ravel())
            else:
                if self.hvg_batch_key is not None and self.hvg_batch_key not in self.adata[0].obs:
                    print(f'Unable to find hvg_batch_key {self.hvg_batch_key} in adata.obs for HVG', flush=True)
                if self.n_top_genes > 0:
                    highly_variable_genes(
                        self.adata, n_top_genes=self.n_top_genes, subset=True,
                        batch_key=self.hvg_batch_key if self.hvg_batch_key in self.adata[0].obs else None,
                    )
                var_names = self.adata[0].var_names.copy()
            for _ in range(len(self.adata)):
                # Subset each adata to feature selected genes
                self.adata[_] = self.adata[_][:, var_names].copy()
        self.var_names = var_names
        self.input_size = len(self.var_names)

        # Encode covariates
        obs_list = [_.obs for _ in self.adata]
        self.counterfactual_categorical_covariates, self.obs_encoding_dict, self.obs_decoding_dict = encode_categorical_covariates(obs_list, self.categorical_covariate_keys, self.unlabeled)
        self.obs_zscoring_dict = encode_continuous_covariates(obs_list, self.continuous_covariate_keys)
        
        # Save number of covariate dimensions
        self.n_categorical_covariate_dims = int(np.sum([max(self.obs_encoding_dict[_].values()) + 1 for _ in self.categorical_covariate_keys]))
        self.n_continuous_covariate_dims = len(self.continuous_covariate_keys)
        self.n_total_covariate_dims = self.n_categorical_covariate_dims + self.n_continuous_covariate_dims

        # Save counterfactual covariates
        self.counterfactual_covariates = np.pad(self.counterfactual_categorical_covariates, (0, self.n_continuous_covariate_dims))
        self.initialized_features = True
        print(f"Encoding covariates with: {self.obs_encoding_dict, self.obs_zscoring_dict}")

        return self.initialized_features

    def prepare_data(self):
        if not self.initialized_features:
            print("Warning: Features not initialized. Calling self.initialize_features()")
            self.initialize_features()

        print('Preparing training data', flush=True)
        self._set_adataset_builder(self.memory_mode)
        train_adatasets = []
        print(f'We have {len(self.adata)} adatas to train on: {self.adata}')
        for adata in self.adata:
            adataset = self._adataset_builder(adata)
            if self.memory_mode == 'backed':
                adataset.set_var_subset(var_subset=self.var_names)
            train_adatasets.append(adataset)
        if len(train_adatasets) > 1:
            print(f"Training on more than one adataset: {train_adatasets}")
            self.train_adataset = ConcatAnnDataset(train_adatasets)
        else:
            print("Training on only one adataset")
            self.train_adataset = train_adatasets[0]
        self.prepared_data = True

        return self.train_adataset

    def prepare_model(self, **model_kwargs):
        if not self.prepared_data:
            print("Warning: Data not prepared. Calling self.prepare_data()")
            self.prepare_data()

        # Update model kwargs with covariate dimensions
        model_kwargs['n_categorical_covariate_dims'] = self.n_categorical_covariate_dims
        model_kwargs['n_total_covariate_dims'] = self.n_total_covariate_dims

        # Update model_kwargs with input and padding dimensions
        # torch.compile may complain if input size is not a multiple of 4
        model_kwargs['input_size'] = self.input_size
        if self.compile_model and self.input_size % 4 != 0:
            model_kwargs['padding_size'] = 4 - (self.input_size % 4)
        else:
            model_kwargs['padding_size'] = 0

        # Initialize model
        match self.distribution:
            case 'nb' | 'zinb':
                self.model = Etude(**model_kwargs)
            case _:
                raise NotImplementedError('ERROR: Only NB and ZINB distributions are currently supported')
        print(
            f'Preparing model with input size: {self.input_size}, distribution: {self.distribution}, '
            f'categorical_covariate_keys: {self.categorical_covariate_keys}, continuous_covariate_keys: {self.continuous_covariate_keys}, '
            f'adversarial: {model_kwargs["adversarial"]}, padding size: {model_kwargs["padding_size"]}'
        )
        self.prepared_model = True

        return self.model

    def deepcopy_model(self):
        assert self.prepared_model

        return copy.deepcopy(self.model)

    def train(self):
        nvtx.range_push(f"Train model")

        optimizer, n_batches, n_samples = self._initialize_training()
        compiled_train_step = self._compile_train_step()
        if self.early_stopping:
            prev_loss = torch.tensor(torch.inf, dtype=torch.float32, device=self.device)
            n_epochs_no_improvement = torch.zeros((), dtype=torch.float32, device=self.device)

        nvtx.range_push(f"Train epoch {0}")
        epoch_losses = {k: torch.zeros((), dtype=torch.float32, device=self.device) for k in self.LOSS_KEYS}
        kld_weight_ = torch.zeros((), dtype=torch.float32, device=self.device)
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
                kld_weight_.fill_(self._get_warmup(epoch_idx, batch_idx, n_batches, self.min_weight, self.max_weight, self.n_annealing_epochs))
                losses_dict = compiled_train_step(self.model, optimizer, batch, kld_weight=kld_weight_, adv_weight=kld_weight_)

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

            self._print_epoch_losses(epoch_losses, kld_weight_)
            self._save_model_checkpoint(epoch_idx)
            if epoch_losses['total'] < best_loss:
                best_epoch = epoch_idx
                best_loss.fill_(epoch_losses['total'])
                best_model_weights = copy.deepcopy(self.model.state_dict())
            nvtx.range_pop()  # "Train epoch {0}"; "Train epoch {epoch_idx + 1}"

            if self._early_stopping(n_epochs_no_improvement, epoch_losses['total'], prev_loss):
                break
            nvtx.range_push(f"Train epoch {epoch_idx + 1}")
        print(f'Training completed for {self.run_name} using device={self.device} and memory_mode={self.memory_mode}', flush=True)
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
        print(f'Training model with up to {self.max_epochs} epochs and random seed: {self.random_seed}', flush=True)

        # Create output directories
        self.checkpoint_path = f'{self.outdir}/checkpoints'
        os.makedirs(f'{self.checkpoint_path}', exist_ok=True)

        # Prepare dataloaders
        nvtx.range_push("Prepare Dataloaders")
        seed_worker, dataloader_generator = self.set_determinism()
        self.train_adata_loader = DataLoader(
            self.train_adataset, batch_size=None, num_workers=self.num_workers,
            sampler=self._get_sampler(self.train_adataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last, memory_mode=self.memory_mode),
            worker_init_fn=seed_worker,
            generator=dataloader_generator,
            persistent_workers = self.num_workers > 0,
            pin_memory=(self.memory_mode not in ('GPU', 'SparseGPU')),  # speeds up host to device copy
        )
        total_cells = len(self.train_adataset)
        if self.drop_last:
            n_batches = total_cells // self.batch_size
            n_samples = n_batches * self.batch_size
        else:
            n_batches = (total_cells + self.batch_size - 1) // self.batch_size
            n_samples = len(self.train_adataset)
        nvtx.range_pop()  # "Prepare Dataloaders"

        nvtx.range_push("Prepare optimizer")
        print(f'Training started for {self.run_name} using device={self.device} and memory_mode={self.memory_mode}', flush=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model = self.model.to(device=self.device)
        self.model.train()
        nvtx.range_pop()  # "Prepare optimizer"

        nvtx.range_push("Save model weights")
        if self.save_initial_weights or self.checkpoint_every_n_epochs is not None:
            torch.save(self.model.state_dict(), f'{self.checkpoint_path}/model_epoch=-1.pt')
            print(f'Model saved at {self.checkpoint_path}/model_epoch=-1.pt', flush=True)
        nvtx.range_pop()  # "Save model weights"

        nvtx.range_pop()  # "Initialize training"

        return optimizer, n_batches, n_samples

    def _compile_train_step(self):
        nvtx.range_push("Compile train step")

        def train_step(model, optimizer, batch, kld_weight, adv_weight):
            # Forward pass
            losses_dict = model.training_step(batch, kld_weight, adv_weight)
            total_ = losses_dict['total']

            # Backward pass
            optimizer.zero_grad()
            total_.backward()
            optimizer.step()

            return losses_dict

        if torch.cuda.is_available() and self.compile_model:
            compiled_train_step = torch.compile(train_step, mode='max-autotune')  # , fullgraph=True)
            print('Model compiling for up to 10x faster training', flush=True)
        else:
            compiled_train_step = train_step
            print('CUDA not available or compilation turned off', flush=True)
        
        nvtx.range_pop()  # "Compile train step"

        return compiled_train_step

    def _print_epoch_losses(self, epoch_losses, kld_weight, flush=False):
        nvtx.range_push("Print epoch losses")
        msg = 'Epoch '
        msg += ', '.join(f"{k.capitalize()}: {epoch_losses[k]:.3f}" for k in self.LOSS_KEYS)
        msg += f", KLD weight: {kld_weight:.6f}"
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
        self.trained_model = True
        nvtx.range_pop()  # "Save model and var_names"

    # ======================
    # Internal utilities
    # ======================
    def _get_adataset(
        self,
        adata=None,
        memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'backed'] | None] = None,
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

        # Update memory mode
        prev_memory_mode = self.memory_mode
        if memory_mode is None:
            memory_mode = self.memory_mode
        self._set_adataset_builder(memory_mode)

        # Update data
        if adata is None:
            print(f"Warning: No adata passed in. Using first adata in self.adata (list of adata)")
            adata = self.adata[0]
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
        shuffle: bool = False,
        drop_last: bool = False,
        memory_mode: Literal['GPU', 'SparseGPU', 'CPU', 'backed'] = None,
    ):
        if memory_mode is None:
            memory_mode = self.memory_mode
        if memory_mode in ('GPU', 'SparseGPU') and torch.cuda.is_available():
            return GPUBatchSampler(
                adataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        else:
            return BatchSampler(
                RandomSampler(adataset) if shuffle else SequentialSampler(adataset),
                batch_size=batch_size,
                drop_last=drop_last,
            )

    def _get_warmup(
        self,
        epoch, batch_idx=0, n_batches=1,
        min_weight=0, max_weight=1.0, n_annealing_epochs=400,
    ):
        training_progress = epoch + (batch_idx / n_batches)

        # Use a linear beta-annealing schedule
        return min_weight + training_progress / n_annealing_epochs * (max_weight - min_weight)

    def _set_adataset_builder(
        self,
        memory_mode: Union[Literal['GPU', 'SparseGPU', 'CPU', 'backed'] | None] = None,
    ):
        """
        Parameters
        ----------
        memory_mode : Union[Literal['GPU', 'SparseGPU', 'CPU', 'backed'] | None], optional
            If None, uses self.memory mode (by default)

        Raises
        ------
        NotImplementedError
            Only supports GPU, SparseGPU, CPU, and backed memory modes
        """

        if memory_mode is None:
            memory_mode = self.memory_mode

        adataset_kwargs = dict(
            memory_mode=memory_mode,
            categorical_covariate_keys=self.categorical_covariate_keys,
            continuous_covariate_keys=self.continuous_covariate_keys,
            obs_encoding_dict=self.obs_encoding_dict,
            obs_decoding_dict=self.obs_decoding_dict,
            obs_zscoring_dict=self.obs_zscoring_dict,
        )

        match memory_mode:
            case 'GPU' | 'CPU':
                self._adataset_builder = partial(AnnDataset, **adataset_kwargs)
            case 'SparseGPU':
                self._adataset_builder = partial(SparseGPUAnnDataset, **adataset_kwargs)
            case 'backed':
                self._adataset_builder = partial(BackedAnnDataset, **adataset_kwargs)
            case _:
                raise NotImplementedError('Only GPU, SparseGPU, CPU, and backed modes are supported')
