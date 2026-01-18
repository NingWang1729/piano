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

# piano/__init__.py

# Define package version
__version__ = '0.1.0'

# Import all modules
from .models.base_models import Etude, ZinbEtude, PaddedEtude, PaddedZinbEtude
from .models.base_modules import GradReverse, grad_reverse
from .utils.composer import Composer
from .utils.covariates import encode_categorical_covariates, encode_continuous_covariates
from .utils.data import AnnDataset, SparseGPUAnnDataset, BackedAnnDataset, GPUBatchSampler, streaming_hvg_indices
from .utils.timer import time_code
from .utils.triton_sparse import SparseTritonMatrix

# Specify all imports (i.e. `from piano import *`)
__all__ = [
    # .models
    ## .base_models
    'Etude',
    'ZinbEtude',
    'PaddedEtude',
    'PaddedZinbEtude',
    ## .base_modules
    'GradReverse',
    'grad_reverse',
    # .utils
    ## .composer
    'Composer',
    ## .covariates
    'encode_categorical_covariates',
    'encode_continuous_covariates',
    ## .data
    'AnnDataset',
    'SparseGPUAnnDataset',
    'BackedAnnDataset',
    'GPUBatchSampler',
    'streaming_hvg_indices',
    ## .timer
    'time_code',
    ## .triton_sparse
    'SparseTritonMatrix',
]
