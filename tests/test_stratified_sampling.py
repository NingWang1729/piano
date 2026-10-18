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

from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from piano import Composer
from piano.utils.data import AnnDataset, TensorBatchSampler

CLASS_SIZES = {'x': 200, 'y': 80, 'z': 20}  # Uneven so a cap of 50 trims x and y but not z
N_GENES = 50


# ---------------------------------------------------------------------------
# Helper: small count matrix with a batch covariate and uneven cell type classes
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    cell_types = np.repeat(list(CLASS_SIZES), list(CLASS_SIZES.values()))
    n_cells = len(cell_types)
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], n_cells)),
        'cell_type': pd.Categorical(cell_types),
    }, index=[str(i) for i in range(n_cells)])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(n_cells, N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


def sample_one_epoch(sampler):
    return torch.cat(list(sampler)).numpy()


def test_sampler_without_stratification_yields_every_cell_once():
    dataset = AnnDataset(make_adata(), memory_mode='CPU')
    sampler = TensorBatchSampler(dataset, batch_size=32, shuffle=True, drop_last=False)
    assert np.array_equal(np.sort(sample_one_epoch(sampler)), np.arange(len(dataset)))


def test_stratified_sampler_caps_each_class():
    dataset = AnnDataset(make_adata(), memory_mode='CPU', stratify_column='cell_type')
    sampler = TensorBatchSampler(dataset, batch_size=32, shuffle=True, drop_last=False, samples_per_class=50)
    indices = sample_one_epoch(sampler)

    # Each class contributes min(cap, class size) distinct cells per epoch
    counts = Counter(dataset.obs['cell_type'].to_numpy()[indices])
    assert counts == {name: min(50, size) for name, size in CLASS_SIZES.items()}
    assert len(np.unique(indices)) == len(indices)


def test_stratify_column_without_samples_per_class_warns(capsys):
    # Training would silently use all cells, so the user is told stratification is off
    Composer(make_adata(), memory_mode='CPU', compile_model=False, stratify_column='cell_type')
    assert "samples_per_class is None" in capsys.readouterr().out


def test_samples_per_class_without_stratify_column_raises():
    # Previously failed with "TypeError: 'NoneType' object is not iterable"
    dataset = AnnDataset(make_adata(), memory_mode='CPU')
    with pytest.raises(ValueError, match='stratify_column'):
        TensorBatchSampler(dataset, batch_size=32, samples_per_class=50)


# ---------------------------------------------------------------------------
# End-to-end: one CPU training epoch must not crash, with the default
# (unstratified) settings and with stratified training
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('stratify_kwargs', [{}, {'stratify_column': 'cell_type', 'samples_per_class': 50}], ids=['default', 'stratified'])
def test_composer_trains_one_epoch(stratify_kwargs, tmp_path):
    pianist = Composer(
        make_adata(), memory_mode='CPU', compile_model=False,
        categorical_covariate_keys=['batch'], n_top_genes=N_GENES,
        max_epochs=1, batch_size=32, outdir=str(tmp_path),
        **stratify_kwargs,
    )
    pianist.run_pipeline()
    assert pianist.trained_model
