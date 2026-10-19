from __future__ import annotations

from functools import singledispatch
from typing import List

import anndata as ad
import numba
import numpy as np
import pandas as pd
from fast_array_utils import stats
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, csc_matrix, csr_array, csc_array
from skmisc.loess import loess

# From ScanPy: https://github.com/scverse/scanpy/blob/main/src/scanpy/_compat.py#L47
CSRBase = csr_matrix | csr_array
CSCBase = csc_matrix | csc_array
CSBase = CSRBase | CSCBase

# skmisc's loess crashes the process (segmentation fault) instead of raising when given fewer points than this
MIN_LOESS_GENES = 4


@singledispatch
def clip_square_sum(
    data_batch: np.ndarray, clip_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Clip data_batch by clip_val.

    Parameters
    ----------
    data_batch
        The data to be clipped
    clip_val
        Clip by these values (must be broadcastable to the input data)

    Returns
    -------
        The clipeed data
    """
    batch_counts = data_batch.astype(np.float64).copy()
    clip_val_broad = np.broadcast_to(clip_val, batch_counts.shape)
    np.putmask(
        batch_counts,
        batch_counts > clip_val_broad,
        clip_val_broad,
    )

    squared_batch_counts_sum = np.square(batch_counts).sum(axis=0)
    batch_counts_sum = batch_counts.sum(axis=0)
    return squared_batch_counts_sum, batch_counts_sum


@clip_square_sum.register(CSBase)
def _(data_batch: CSBase, clip_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch_counts = data_batch if isinstance(data_batch, CSRBase) else data_batch.tocsr()

    return _sum_and_sum_squares_clipped(
        batch_counts.indices,
        batch_counts.data,
        n_cols=batch_counts.shape[1],
        clip_val=clip_val,
        nnz=batch_counts.nnz,
    )

# parallel=False needed for accuracy
@numba.njit(cache=True, parallel=False)
def _sum_and_sum_squares_clipped(
    indices: NDArray[np.integer],
    data: NDArray[np.floating],
    *,
    n_cols: int,
    clip_val: NDArray[np.float64],
    nnz: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    squared_batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    batch_counts_sum = np.zeros(n_cols, dtype=np.float64)
    for i in numba.prange(nnz):
        idx = indices[i]
        element = min(np.float64(data[i]), clip_val[idx])
        squared_batch_counts_sum[idx] += element**2
        batch_counts_sum[idx] += element

    return squared_batch_counts_sum, batch_counts_sum


def _check_same_var_names(adata_list: List[ad.AnnData]) -> None:
    """Raise if the AnnDatas do not have the same genes in the same order, since their statistics are combined by position."""
    for index, adata in enumerate(adata_list[1:], start=1):
        if not adata.var_names.equals(adata_list[0].var_names):
            raise ValueError(
                f"adata {index} has different var_names (or a different gene order) than adata 0. "
                f"Highly variable genes combine gene statistics across AnnDatas by position, so all must have the same genes "
                f"in the same order: subset each to the same list of genes, e.g. adata = adata[:, genes].copy()"
            )


def _check_batch_labels_present(adata_list: List[ad.AnnData], batch_key: str) -> None:
    """Raise if a cell has no batch label (NaN or None), since cells are grouped by sorted batch labels."""
    for index, adata in enumerate(adata_list):
        n_missing = adata.obs[batch_key].isna().sum()
        if n_missing > 0:
            raise ValueError(
                f"adata {index}.obs['{batch_key}'] has {n_missing} missing (NaN) batch labels: "
                f"fill them with a batch label or remove those cells before selecting highly variable genes"
            )


def _check_batches_have_two_cells(adata_list: List[ad.AnnData], batch_key: str) -> None:
    """Raise if a batch has fewer than two cells, since its sample variance is undefined."""
    labels = np.concatenate([adata.obs[batch_key].to_numpy() for adata in adata_list])
    n_cells = pd.Series(labels).value_counts()
    too_small = n_cells[n_cells < 2]
    if len(too_small) > 0:
        raise ValueError(
            f"Batch {too_small.index[0]!r} of batch_key '{batch_key}' has only {too_small.iloc[0]} cell, but highly variable "
            f"genes need at least 2 cells per batch: remove that batch's cells or merge it into another batch"
        )


def _check_enough_variable_genes(non_zero: np.ndarray, cells: str) -> None:
    """Raise if too few genes vary across cells (described by cells) to fit the LOESS mean-variance trend."""
    n_variable = np.count_nonzero(non_zero)
    if n_variable < MIN_LOESS_GENES:
        raise ValueError(
            f"Only {n_variable} genes have nonzero variance in {cells}, but fitting the mean-variance trend for highly "
            f"variable genes needs at least {MIN_LOESS_GENES}: remove or merge that batch, or select genes another way"
        )


def highly_variable_genes(
    adata_list: List[ad.AnnData] | ad.AnnData,
    n_top_genes: int = 2000,
    batch_key: str | None = None,
    span: float = 0.3,
    subset: bool = False,
    inplace: bool = True,
    chunk_size: int = 10_000,
) -> pd.DataFrame | None:
    """See `highly_variable_genes`.
    Modified from scanpy.pp.highly_variable_genes (Seurat V3)
    For further implementation details see https://www.overleaf.com/read/ckptrbgzzzpg

    adata_list may also hold AnnDatas read with read_h5ad(path, backed='r'): their counts stay on disk and
    are read in contiguous chunks of chunk_size cells. Backed AnnDatas cannot be changed in place, so inplace
    and subset must be False. Summing in chunks changes results only by floating-point rounding.

    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pd.DataFrame`) or
    updates `.var` with the following fields:

    highly_variable : :class:`bool`
        boolean indicator of highly-variable genes.
    **means**
        means per gene.
    **variances**
        variance per gene.
    **variances_norm**
        normalized variance per gene, averaged in the case of multiple batches.
    highly_variable_rank : :class:`float`
        Rank of the gene according to normalized variance, median rank in the case of multiple batches.
    highly_variable_nbatches : :class:`int`
        If batch_key is given, this denotes in how many batches genes are detected as HVG.

    """
    # Support passing in an individual AnnData
    if not isinstance(adata_list, list):
        adata_list = [adata_list]
    if any(adata.isbacked for adata in adata_list) and (inplace or subset):
        raise ValueError("Backed AnnDatas cannot be changed in place: use inplace=False and subset=False, and select genes from the returned DataFrame")

    _check_same_var_names(adata_list)
    if batch_key is not None:
        _check_batch_labels_present(adata_list, batch_key)
        _check_batches_have_two_cells(adata_list, batch_key)

    df = pd.DataFrame(index=adata_list[0].var_names)
    df["means"], df["variances"], obs = mean_var_across_datas(_iter_counts([_counts(adata, None, chunk_size) for adata in adata_list]))

    # Determine all batch labels across all adatas
    if batch_key is None:
        all_batches = np.array([0])
    else:
        all_batches = np.unique(
            np.concatenate(
                [adata.obs[batch_key].to_numpy() for adata in adata_list]
            )
        )

    norm_gene_vars = []
    for b in all_batches:
        # Retrieve data for corresponding batch
        datas = []
        for adata_i in adata_list:
            if batch_key is None:
                mask = slice(None)
            else:
                mask = adata_i.obs[batch_key].to_numpy() == b
            if np.any(mask):
                datas.append(_counts(adata_i, mask, chunk_size))
        if len(datas) == 0:
            continue

        # === distributed mean / var ===
        mean, var, n_obs = mean_var_across_datas(_iter_counts(datas))
        non_zero = var > 0
        _check_enough_variable_genes(non_zero, "all cells" if batch_key is None else f"batch {b!r} of batch_key '{batch_key}'")
        estimat_var = np.zeros_like(var, dtype=np.float64)

        # Run LOESS smoothing
        x, y = np.log10(mean[non_zero]), np.log10(var[non_zero])
        model = loess(x, y, span=span, degree=2)
        model.fit()
        estimat_var[non_zero] = model.outputs.fitted_values
        reg_std = np.sqrt(10 ** estimat_var)

        # Clip larger gene counts to mu + std * sqrt(num_cells) (as in Seurat V3)
        clip_val = mean + reg_std * np.sqrt(n_obs)
        squared_batch_counts_sum, batch_counts_sum = np.zeros_like(mean, dtype=np.float64), np.zeros_like(mean, dtype=np.float64)
        for data in _iter_counts(datas):
            squared_batch_counts_sum_addend, batch_counts_sum_addend = clip_square_sum(data, clip_val)
            squared_batch_counts_sum += squared_batch_counts_sum_addend
            batch_counts_sum += batch_counts_sum_addend
        norm_gene_var = (
            (n_obs * mean ** 2 + squared_batch_counts_sum - 2 * batch_counts_sum * mean)
            / ((n_obs - 1) * reg_std ** 2)
        )
        norm_gene_vars.append(norm_gene_var[None, :])

    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    # Argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)

    # This is done in SelectIntegrationFeatures() in Seurat V3
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum(
        (ranked_norm_gene_vars < n_top_genes).astype(int), axis=0
    )
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)

    df = df.assign(
        gene_name=df.index,
        highly_variable_nbatches=num_batches_high_var,
        highly_variable_rank=median_ranked,
        variances_norm=np.mean(norm_gene_vars, axis=0),
    )
    sort_cols = ["highly_variable_rank", "highly_variable_nbatches"]
    sort_ascending = [True, False]
    sorted_index = df[sort_cols].sort_values(sort_cols, ascending=sort_ascending, na_position="last").index
    df["highly_variable"] = False
    df.loc[sorted_index[: int(n_top_genes)], "highly_variable"] = True

    if inplace:
        for adata in adata_list:
            adata.uns["hvg"] = {"flavor": "seurat_v3"}
            for _ in ["highly_variable", "highly_variable_rank", "means", "variances"]:
                adata.var[_] = df[_].to_numpy()
            adata.var["variances_norm"] = (df["variances_norm"].to_numpy().astype("float64", copy=False))
            if batch_key is not None:
                adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"].to_numpy()
            if subset:
                adata._inplace_subset_var(df["highly_variable"].to_numpy())
    else:
        if batch_key is None:
            df = df.drop(["highly_variable_nbatches"], axis=1)
        if subset:
            df = df.iloc[df["highly_variable"].to_numpy(), :]

        return df
    return None

class _BackedCounts:
    """Counts of some cells of a backed AnnData, read from disk in contiguous chunks of chunk_size cells."""
    def __init__(self, adata: ad.AnnData, mask: np.ndarray, chunk_size: int):
        self.adata = adata
        self.mask = mask  # Boolean, one value per cell
        self.chunk_size = chunk_size

    def __iter__(self):
        # Contiguous reads are fast on disk; the cells to keep are then selected in memory
        for start in range(0, self.adata.n_obs, self.chunk_size):
            keep = self.mask[start:start + self.chunk_size]
            if keep.any():
                yield self.adata.X[start:start + self.chunk_size][keep]


def _counts(adata, mask, chunk_size):
    """Counts of the cells in mask (None for all, a slice or a boolean array): the in-memory matrix, or for a backed AnnData, chunks read from disk."""
    if adata.isbacked:
        keep = np.zeros(adata.n_obs, dtype=bool)
        keep[slice(None) if mask is None else mask] = True
        return _BackedCounts(adata, keep, chunk_size)
    return adata.X if mask is None else adata.X[mask]


def _iter_counts(datas):
    """Each in-memory matrix as is, and each backed selection chunk by chunk."""
    for data in datas:
        if isinstance(data, _BackedCounts):
            yield from data
        else:
            yield data


def mean_var_across_datas(
    datas: list[ad.AnnData.X],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute per-gene mean and sample variance across multiple AnnData objects
    without concatenation.
    """
    means = []
    variances = []
    n_cells = []
    for data in datas:
        mean_k, var_k = stats.mean_var(data, axis=0, correction=1)
        means.append(mean_k)
        variances.append(var_k)
        n_cells.append(data.shape[0])

    means = np.stack(means)          # (K, G)
    variances = np.stack(variances)  # (K, G)
    n_cells = np.asarray(n_cells)    # (K,)
    global_n_cells = n_cells.sum()

    # Law of iterated expectations
    # E[X] = E[E[X|Y]]
    # E[X] = Sum_y(pmf_x(Y=y) * mean_x(Y=y))
    global_mean = (n_cells[:, None] * means).sum(axis=0) / global_n_cells

    """
    Since we are using sample variances, we use ANOVA (sum-of-squares) identity
        sigma^2 = 1/(n - 1) * Sum_i((X_i - mu)^2)
        = 1/(n - 1) * Sum_k(Sum_i_in_k((X_i - mu)^2))  # Partition over k datasets
    Observe:
        Sum_i_in_k((X_i - mu)^2)
        = Sum_i_in_k(((X_i - mu_k) + (mu_k - mu))^2)
        = Sum_i_in_k((X_i - mu_k)^2 + (mu_k - mu)^2 + 2 * (X_i - mu_k) * (mu_k - mu))
            where Sum_i_in_k(X_i - mu_k) = Sum_i_in_k(X_i) - n_k * mu_k = 0
        = Sum_i_in_k((X_i - mu_k)^2 + (mu_k - mu)^2)
    Rearranging terms:
        sigma_k^2 = 1/(n_k - 1) * Sum_i_in_k((X_i - mu_k)^2)
        Sum_i_in_k((X_i - mu_k)^2) = (n_k - 1) * sigma_k^2
    Therefore:
        sigma^2
        = 1/(n - 1) * Sum_k(Sum_i_in_k((X_i - mu_k)^2 + (mu_k - mu)^2))
        = 1/(n - 1) * Sum_k((n_k - 1) * sigma_k^2 + n_k * (mu_k - mu)^2))
    """
    if global_n_cells <= 1:
        raise ValueError(f"A sample variance needs at least 2 cells, but only {global_n_cells} were found across all datasets")
    within = ((n_cells - 1)[:, None] * variances).sum(axis=0)  # Shape (G,)
    between = (n_cells[:, None] * (means - global_mean) ** 2).sum(axis=0)  # Shape (G,)
    global_var = 1 / (global_n_cells - 1) * (within + between)  # Shape (G,)

    return global_mean, global_var, global_n_cells

def highly_variable_genes_1_adata(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    batch_key: str | None = None,
    span: float = 0.3,
    subset: bool = False,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """See `highly_variable_genes`.
    Modified from scanpy.pp.highly_variable_genes (Seurat V3)
    For further implementation details see https://www.overleaf.com/read/ckptrbgzzzpg

    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pd.DataFrame`) or
    updates `.var` with the following fields:

    highly_variable : :class:`bool`
        boolean indicator of highly-variable genes.
    **means**
        means per gene.
    **variances**
        variance per gene.
    **variances_norm**
        normalized variance per gene, averaged in the case of multiple batches.
    highly_variable_rank : :class:`float`
        Rank of the gene according to normalized variance, median rank in the case of multiple batches.
    highly_variable_nbatches : :class:`int`
        If batch_key is given, this denotes in how many batches genes are detected as HVG.

    """
    df = pd.DataFrame(index=adata.var_names)
    data = adata.X

    df["means"], df["variances"] = stats.mean_var(data, axis=0, correction=1)

    batch_info = (
        pd.Categorical(np.zeros(adata.shape[0], dtype=int))
        if batch_key is None
        else adata.obs[batch_key].to_numpy()
    )

    norm_gene_vars = []
    for b in np.unique(batch_info):
        data_batch = data[batch_info == b]

        mean, var = stats.mean_var(data_batch, axis=0, correction=1)
        non_zero = var > 0
        estimat_var = np.zeros(data.shape[1], dtype=np.float64)

        # Run LOESS smoothing
        y = np.log10(var[non_zero])
        x = np.log10(mean[non_zero])
        model = loess(x, y, span=span, degree=2)
        model.fit()
        estimat_var[non_zero] = model.outputs.fitted_values
        reg_std = np.sqrt(10 ** estimat_var)

        # Clip larger gene counts to mu + std * sqrt(num_cells) (as in Seurat V3)
        n_obs = data_batch.shape[0]
        clip_val = mean + reg_std * np.sqrt(n_obs)
        squared_batch_counts_sum, batch_counts_sum = clip_square_sum(data_batch, clip_val)
        norm_gene_var = (1 / ((n_obs - 1) * np.square(reg_std))) * (
            (n_obs * np.square(mean))
            + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )
        norm_gene_vars.append(norm_gene_var.reshape(1, -1))

    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    # argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)

    # this is done in SelectIntegrationFeatures() in Seurat v3
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum(
        (ranked_norm_gene_vars < n_top_genes).astype(int), axis=0
    )
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)

    df = df.assign(
        gene_name=df.index,
        highly_variable_nbatches=num_batches_high_var,
        highly_variable_rank=median_ranked,
        variances_norm=np.mean(norm_gene_vars, axis=0),
    )
    sort_cols = ["highly_variable_rank", "highly_variable_nbatches"]
    sort_ascending = [True, False]
    sorted_index = df[sort_cols].sort_values(sort_cols, ascending=sort_ascending, na_position="last").index
    df["highly_variable"] = False
    df.loc[sorted_index[: int(n_top_genes)], "highly_variable"] = True

    if inplace:
        adata.uns["hvg"] = {"flavor": "seurat_v3"}
        for _ in ["highly_variable", "highly_variable_rank", "means", "variances"]:
            adata.var[_] = df[_].to_numpy()
        adata.var["variances_norm"] = (df["variances_norm"].to_numpy().astype("float64", copy=False))
        if batch_key is not None:
            adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"].to_numpy()
        if subset:
            adata._inplace_subset_var(df["highly_variable"].to_numpy())
    else:
        if batch_key is None:
            df = df.drop(["highly_variable_nbatches"], axis=1)
        if subset:
            df = df.iloc[df["highly_variable"].to_numpy(), :]

        return df
    return None

def highly_variable_genes_batch_streaming(
    adata_paths: List[str] | str,
    n_top_genes: int = 2000,
    span: float = 0.3,
) -> pd.DataFrame | None:
    """See `highly_variable_genes`.
    Modified from scanpy.pp.highly_variable_genes (Seurat V3)
    For further implementation details see https://www.overleaf.com/read/ckptrbgzzzpg
    This implementation supports streaming HVG calculations over multiple paths.
    Specifically, each adata contains all cells from a unique batch.
    This is designed for large datasets, where each batch to integrate is stored in its own file.

    Returns
    -------
    Dataframe of highly variable genes:

    highly_variable : :class:`bool`
        boolean indicator of highly-variable genes.
    **variances_norm**
        normalized variance per gene, averaged in the case of multiple batches.
    highly_variable_rank : :class:`float`
        Rank of the gene according to normalized variance, median rank in the case of multiple batches.
    highly_variable_nbatches : :class:`int`
        If batch_key is given, this denotes in how many batches genes are detected as HVG.

    """
    # Support passing in an individual AnnData
    if not isinstance(adata_paths, list):
        adata_paths = [adata_paths]
    if not isinstance(n_top_genes, list):
        n_top_genes = [n_top_genes]

    var_names = None
    norm_gene_vars = []
    for adata_path in adata_paths:
        # Retrieve data for corresponding batch. Each adata contains all and only cells from a unique batch.
        adata = sc.read_h5ad(adata_path)
        n_obs = adata.X.shape[0]
        if n_obs <= 1:
            print(f"Warning: {adata_path} contains <= 1 cell, skipping")
            continue
        if var_names is None:
            var_names = adata.var_names
        elif not np.array_equal(var_names, adata.var_names):
            print(f"Warning: All AnnData objects should have identical var_names: {adata_path} is different!")
        
        # Calculate summary statistics
        mean, var = stats.mean_var(adata.X, axis=0, correction=1)
        non_zero = var > 0
        if (non_zero <= 0).all():
            print(f"Warning: {adata_path} contains all zero counts, skipping")
            continue  # or handle explicitly
        estimat_var = np.zeros(adata.X.shape[1], dtype=np.float64)

        # Run LOESS smoothing
        x, y = np.log10(mean[non_zero]), np.log10(var[non_zero])
        model = loess(x, y, span=span, degree=2)
        model.fit()
        estimat_var[non_zero] = model.outputs.fitted_values
        reg_std = np.sqrt(10 ** estimat_var)

        # Clip larger gene counts to mu + std * sqrt(num_cells) (as in Seurat V3)
        clip_val = mean + reg_std * np.sqrt(n_obs)
        squared_batch_counts_sum, batch_counts_sum = clip_square_sum(adata.X, clip_val)
        norm_gene_var = (n_obs * np.square(mean) + squared_batch_counts_sum - 2 * batch_counts_sum * mean) / ((n_obs - 1) * np.square(reg_std))
        norm_gene_vars.append(norm_gene_var.reshape(1, -1))
        
        # Free up memory
        del adata
    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    
    # Argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)
    # This is done in SelectIntegrationFeatures() in Seurat V3
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    df = pd.DataFrame(index=var_names)
    df = df.assign(
        gene_name=df.index,
        variances_norm=np.mean(norm_gene_vars, axis=0),
    )
    for n_genes in n_top_genes:
        if n_genes <= 0:
            n_genes = len(var_names)
        ranked_norm_gene_vars_copy = ranked_norm_gene_vars.copy()
        num_batches_high_var = np.sum((ranked_norm_gene_vars_copy < n_genes).astype(int), axis=0)
        ranked_norm_gene_vars_copy[ranked_norm_gene_vars_copy >= n_genes] = np.nan
        ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars_copy)
        median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)
        df[f'highly_variable_rank__{n_genes}'] = median_ranked
        df[f'highly_variable_nbatches__{n_genes}'] = num_batches_high_var
        sort_cols = [f'highly_variable_rank__{n_genes}', f'highly_variable_nbatches__{n_genes}']
        sort_ascending = [True, False]
        sorted_index = df[sort_cols].sort_values(sort_cols, ascending=sort_ascending, na_position="last").index
        df[f"highly_variable__{n_genes}"] = False
        df.loc[sorted_index[: n_genes], f"highly_variable__{n_genes}"] = True

    # Add per-batch normalized variances as separate columns
    for batch, batch_var in zip(adata_paths, norm_gene_vars):
        df[f"var_batch__{batch}"] = batch_var

    return df
