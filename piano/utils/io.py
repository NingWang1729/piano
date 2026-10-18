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

from pathlib import Path

import anndata as ad
import h5py


def read_adata(path) -> ad.AnnData:
    """
    Read a .h5ad, .zarr or .loom file fully into memory as an AnnData, choosing the reader by file extension.

    Composer takes AnnData objects only; the command line interface reads its input files here, so support for
    other formats (or faster readers of these) can be added in one place.
    """
    suffix = Path(path).suffix.lower()
    match suffix:
        case '.h5ad':
            return ad.read_h5ad(path)
        case '.zarr':
            return ad.read_zarr(path)
        case '.loom':
            return read_loom(path)
        case _:
            raise ValueError(f"Cannot read '{path}': piano's command line interface reads .h5ad, .zarr and .loom files")


def read_loom(path) -> ad.AnnData:
    """
    Read a .loom file with anndata's read_loom, which needs the optional loompy package (piano-integration[loom]).

    Cell and gene names are taken from the obs_names/var_names attributes that anndata's write_loom stores,
    or else from loom's standard CellID/Gene attributes used by other tools.
    """
    try:
        import loompy  # noqa: F401  (anndata's read_loom imports it)
    except ImportError:
        raise ImportError("Reading .loom files needs the optional loompy package: pip install 'piano-integration[loom]' (or pip install loompy)") from None
    with h5py.File(path, 'r') as f:
        obs_names = 'obs_names' if 'obs_names' in f['col_attrs'] else 'CellID'
        var_names = 'var_names' if 'var_names' in f['row_attrs'] else 'Gene'
    return ad.read_loom(path, obs_names=obs_names, var_names=var_names)
