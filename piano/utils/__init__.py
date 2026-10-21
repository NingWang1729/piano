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

# piano/utils/__init__.py

import importlib

# Import all modules
from .composer import Composer
from .covariates import encode_categorical_covariates, encode_continuous_covariates
from .data import AnnDataset, SparseGPUAnnDataset, SparseCPUAnnDataset, BackedAnnDataset, ConcatAnnDataset, TensorBatchSampler, streaming_hvg_indices
from .preprocessing import highly_variable_genes, read_gene_set_genes
from .timer import time_code
from .triton_sparse import SparseTritonMatrix

# The UMAP and KNN helpers import umap, pynndescent and faiss, which take several seconds to load
# (much longer on a cold cache, when numba compiles), so they are imported on first use instead
_LAZY_IMPORTS = {
    'faiss_knn': '.umap',
    'faiss_knn_to_umap_graph': '.umap',
    'pca_init': '.umap',
    'optimize_embedding': '.umap',
    'faiss_umap': '.umap',
    'plot_umaps': '.umap',
    'FAISS_KNN': '.knn',
}


def __getattr__(name):
    """Import the lazily loaded names in _LAZY_IMPORTS on first access (PEP 562)."""
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Specify all imports (i.e., `from piano.utils import *`)
__all__ = [
    # .utils
    ## .composer
    'Composer',
    ## .covariates
    'encode_categorical_covariates',
    'encode_continuous_covariates',
    ## .data
    'AnnDataset',
    'SparseGPUAnnDataset',
    'SparseCPUAnnDataset',
    'BackedAnnDataset',
    'ConcatAnnDataset',
    'TensorBatchSampler',
    'streaming_hvg_indices',
    ## .preprocessing
    'highly_variable_genes',
    'read_gene_set_genes',
    ## .timer
    'time_code',
    ## .triton_sparse
    'SparseTritonMatrix',
    ## .umap
    'faiss_knn',
    'faiss_knn_to_umap_graph',
    'pca_init',
    'optimize_embedding',
    'faiss_umap',
    'plot_umaps',
    ## .knn
    'FAISS_KNN',
]
