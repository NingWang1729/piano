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

# triton_sparse.py  ─── Memory-efficient CSR → dense row expander
# -------------------------------------------------------------------
import numpy as np
import torch
import triton
import triton.language as tl
from torch import Tensor
from scipy.sparse import csr_matrix


# -------------------------------------------------------------------
# 1 ▪ Single Triton kernel, parametrised by constexpr flags
# -------------------------------------------------------------------
@triton.jit
def _output_row_offset(m, ncols: tl.constexpr):
    # Start of row m in the dense (M, ncols) output, in elements.
    # m comes from tl.program_id, which is int32, so m * ncols would overflow once the offset
    # reaches 2^31 (e.g. 32770 rows of 65535 genes) and write before the output buffer.
    # Computing it in int64 avoids that on both the int32 and the int64 indptr paths.
    return m.to(tl.int64) * ncols


@triton.jit
def _csr_expand_kernel(
    out_ptr, row_starts_ptr, row_nnz_ptr,
    idx_ptr, val_ptr,
    ncols: tl.constexpr,
    BLOCK: tl.constexpr,
    _ptr_is_i64: tl.constexpr,
    _inds_are_u16: tl.constexpr,
    _inds_are_i16: tl.constexpr,
):
    m = tl.program_id(0)                  # one program ⇒ one row

    # ---- row metadata ----------------------------------------------------
    if tl.constexpr(_ptr_is_i64):
        nnz  = tl.load(row_nnz_ptr   + m)          # int64
        beg  = tl.load(row_starts_ptr + m)         # int64
        i    = tl.zeros((), dtype=tl.int64)
        offs = tl.arange(0, BLOCK).to(tl.int64)
    else:
        nnz  = tl.load(row_nnz_ptr   + m)          # int32
        beg  = tl.load(row_starts_ptr + m)         # int32
        i    = tl.zeros((), dtype=tl.int32)
        offs = tl.arange(0, BLOCK)                 # int32

    # ---- iterate through nnz --------------------------------------------
    while i < nnz:
        cur   = i + offs
        mask  = cur < nnz
        csr_i = beg + cur

        # ---- load column indices, cast to int32 -------------------------
        if tl.constexpr(_inds_are_u16):
            cols16 = tl.load(idx_ptr + csr_i, mask=mask, other=0)   # uint16
            cols32 = cols16.to(tl.int32)

        elif tl.constexpr(_inds_are_i16):
            cols16 = tl.load(idx_ptr + csr_i, mask=mask, other=0)   # int16
            cols32 = cols16.to(tl.int32)

        else:                           # indices stored as int32 or int64
            cols = tl.load(idx_ptr + csr_i, mask=mask, other=0)     # i32/i64
            cols32 = cols.to(tl.int32)   # safe because ncols < 2³¹

        # ---- load data & scatter ----------------------------------------
        vals = tl.load(val_ptr + csr_i, mask=mask, other=0)
        tl.store(out_ptr + _output_row_offset(m, ncols) + cols32, vals, mask=mask)

        i += BLOCK


# -------------------------------------------------------------------
# 2 ▪ Python launcher
# -------------------------------------------------------------------
def expand_rows(
    indptr: Tensor,
    indices: Tensor,
    data: Tensor,
    which_rows: Tensor,
    ncols: int,
    block: int = 128,
) -> Tensor:
    row_starts = indptr[which_rows].contiguous()
    row_nnz    = (indptr[which_rows + 1] - row_starts).contiguous()

    M   = which_rows.numel()
    out = torch.zeros((M, ncols), dtype=data.dtype, device=data.device)

    _csr_expand_kernel[(M,)](
        out, row_starts, row_nnz,
        indices.contiguous(), data,
        ncols, BLOCK=block,
        _ptr_is_i64   = indptr.dtype  == torch.int64,
        _inds_are_u16 = indices.dtype == torch.uint16,
        _inds_are_i16 = indices.dtype == torch.int16,
    )
    return out


# -------------------------------------------------------------------
# 3 ▪ Convenience wrapper with smart dtype-picking
# -------------------------------------------------------------------
class SparseTritonMatrix:
    """GPU CSR with automatic int-width minimisation."""
    def __init__(self, X: csr_matrix, device="cuda"):
        assert isinstance(X, csr_matrix)
        self.shape = X.shape
        self.ncols = X.shape[1]

        nnz_total    = X.indptr[-1]
        indptr_dtype = torch.int32 if nnz_total < 2**31 else torch.int64

        if self.ncols <= 0xFFFF:
            idx_dtype = torch.uint16 if hasattr(torch, "uint16") else torch.int16
        elif self.ncols < 2**31:
            idx_dtype = torch.int32
        else:
            idx_dtype = torch.int64

        self.indptr  = torch.as_tensor(X.indptr,  dtype=indptr_dtype, device=device)
        self.indices = torch.as_tensor(X.indices, dtype=idx_dtype,    device=device)
        self.data    = torch.as_tensor(X.data,                      device=device)

    # --- dense row materialisation ---------------------------------------
    def __getitem__(self, rows):
        if isinstance(rows, (list, np.ndarray)):
            rows = torch.as_tensor(rows, dtype=torch.int64, device=self.data.device)
        return expand_rows(self.indptr, self.indices, self.data, rows, self.ncols)

    # --- convenience ------------------------------------------------------
    def size_bytes(self):
        return (
            self.indptr.element_size()  * self.indptr.numel() +
            self.indices.element_size() * self.indices.numel() +
            self.data.element_size()    * self.data.numel()
        )


# -------------------------------------------------------------------
# 4 ▪ Simple correctness & memory test
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    import scanpy as sc
    import tqdm

    file_path = sys.argv[1]
    cts = sc.read_h5ad(file_path)

    print(f"Sparse matrix shape: {cts.X.shape}")
    sm = SparseTritonMatrix(cts.X)
    print(f"GPU CSR footprint: {sm.size_bytes() / 1024**2:.2f} MB")
    print(f"Indptr dtype: {sm.indptr.dtype}")
    print(f"Indices dtype: {sm.indices.dtype}")
    print(f"Data dtype: {sm.data.dtype}")

    for _ in tqdm.tqdm(range(10_000)):
        rows = np.random.randint(0, cts.X.shape[0], size=1024)
        dense = sm[rows]
        # assert dense.sum().cpu().item() == cts.X[rows, :].sum()
