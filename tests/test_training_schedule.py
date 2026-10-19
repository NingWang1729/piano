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

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from piano import Composer
from piano.utils.data import TensorBatchSampler

CLASS_SIZES = {'x': 200, 'y': 80, 'z': 20}  # Uneven, so a cap of 50 cells per class trims x and y but not z
N_GENES = 50


# ---------------------------------------------------------------------------
# Helper: small count matrix with a batch covariate and uneven cell type classes
# ---------------------------------------------------------------------------
def make_adata(seed=0):
    rng = np.random.default_rng(seed)
    cell_types = np.repeat(list(CLASS_SIZES), list(CLASS_SIZES.values()))
    obs = pd.DataFrame({
        'batch': pd.Categorical(rng.choice(['a', 'b'], len(cell_types))),
        'cell_type': pd.Categorical(cell_types),
    }, index=[str(i) for i in range(len(cell_types))])
    adata = ad.AnnData(X=csr_matrix(rng.poisson(1.0, size=(len(cell_types), N_GENES)).astype(np.float32)), obs=obs)
    adata.var_names = [f'gene_{i}' for i in range(N_GENES)]
    return adata


@pytest.mark.parametrize('epoch, expected', [(0, 0.0), (100, 0.125), (200, 0.25), (300, 0.25), (399, 0.25)])
def test_annealed_weight_stops_at_its_maximum(epoch, expected):
    # Annealing over 200 epochs from 0 to 0.25; training may run longer, e.g. --max_epochs 400 with the CLI's default 200
    pianist = Composer(make_adata(), memory_mode='CPU', compile_model=False, n_top_genes=-1, deterministic=False)
    weight = pianist._get_warmup(epoch, batch_idx=0, n_batches=10, min_weight=0.0, max_weight=0.25, n_annealing_epochs=200)
    # Previously the weight kept growing after annealing (0.499 at epoch 399)
    assert weight == pytest.approx(expected)


@pytest.mark.parametrize('stratified, drop_last, expected', [
    (False, True, (9, 288)),    # 300 cells in batches of 32, last partial batch dropped
    (False, False, (10, 300)),
    (True, True, (3, 96)),      # 50 + 50 + 20 = 120 cells per epoch with a cap of 50 per class
    (True, False, (4, 120)),
], ids=['all_cells', 'all_cells_keep_last', 'stratified', 'stratified_keep_last'])
def test_epoch_counts_follow_the_sampler(stratified, drop_last, expected):
    # n_batches drives annealing progress and n_samples turns summed losses into per-cell losses
    stratification = {'stratify_column': 'cell_type', 'samples_per_class': 50} if stratified else {}
    pianist = Composer(make_adata(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'], n_top_genes=-1,
                       batch_size=32, drop_last=drop_last, deterministic=False, **stratification)
    pianist.initialize_features()
    pianist.prepare_data()
    pianist.prepare_model(**pianist.model_kwargs)
    _, n_batches, n_samples = pianist._initialize_training()
    # Previously these came from the total cell count, so stratified epochs were counted as 300 cells
    assert (n_batches, n_samples) == expected
    assert n_batches == len(list(pianist.train_adata_loader))


@pytest.mark.parametrize('drop_last, expected', [(True, 0), (False, 1)])
def test_sampler_yields_no_empty_batch(drop_last, expected):
    # 10 cells and batches of 32: dropping the partial batch leaves none
    sampler = TensorBatchSampler(list(range(10)), batch_size=32, shuffle=False, drop_last=drop_last)
    batches = list(sampler)
    # Previously yielded one empty batch with drop_last=True, though len(sampler) was 0
    assert len(batches) == len(sampler) == expected
    assert all(len(batch) > 0 for batch in batches)


@pytest.mark.parametrize('stratification, n_cells_per_epoch', [
    ({}, 20),
    ({'stratify_column': 'cell_type', 'samples_per_class': 5}, 5),
], ids=['all_cells', 'stratified'])
def test_composer_without_batches_raises_when_created(stratification, n_cells_per_epoch, tmp_path):
    # 20 cells (all of type x), or 5 with 5 per class, are fewer than one batch of 32, which drop_last=True drops.
    # Previously accepted until training, which failed with "ZeroDivisionError: division by zero" in the annealing schedule
    with pytest.raises(ValueError, match=f'Training would have no batches: an epoch draws {n_cells_per_epoch} cells'):
        Composer(make_adata()[:20].copy(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'],
                 n_top_genes=-1, batch_size=32, drop_last=True, deterministic=False, outdir=str(tmp_path), **stratification)


def test_stratified_classes_are_counted_across_anndatas(tmp_path):
    # Two AnnDatas with 3 cells of type x each: the training dataset has one class of 6 cells, so 5 per class gives
    # 5 cells, fewer than a batch of 6 (counting each AnnData's class on its own would give 3 + 3 = 6 cells, one batch)
    with pytest.raises(ValueError, match='Training would have no batches: an epoch draws 5 cells'):
        Composer([make_adata()[:3].copy(), make_adata(seed=1)[:3].copy()], memory_mode='CPU', compile_model=False,
                 categorical_covariate_keys=['batch'], n_top_genes=-1, batch_size=6, drop_last=True, deterministic=False,
                 stratify_column='cell_type', samples_per_class=5, outdir=str(tmp_path))


@pytest.mark.parametrize('n_cells, drop_last', [(32, True), (20, False)])
def test_composer_with_one_batch_trains(n_cells, drop_last, tmp_path):
    # Exactly one full batch, or one incomplete batch that drop_last=False keeps
    pianist = Composer(make_adata()[:n_cells].copy(), memory_mode='CPU', compile_model=False, categorical_covariate_keys=['batch'],
                       n_top_genes=-1, batch_size=32, max_epochs=1, drop_last=drop_last, deterministic=False, outdir=str(tmp_path))
    pianist.run_pipeline()
    assert pianist.trained_model

