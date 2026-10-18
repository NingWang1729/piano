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


def read_adata(path) -> ad.AnnData:
    """
    Read a .h5ad or .zarr file fully into memory as an AnnData, choosing the reader by file extension.

    Composer takes AnnData objects only; the command line interface reads its input files here, so support for
    other formats (or faster readers of these) can be added in one place.
    """
    suffix = Path(path).suffix.lower()
    match suffix:
        case '.h5ad':
            return ad.read_h5ad(path)
        case '.zarr':
            return ad.read_zarr(path)
        case _:
            raise ValueError(f"Cannot read '{path}': piano's command line interface reads .h5ad and .zarr files")
