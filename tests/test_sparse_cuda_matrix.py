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

# test_triton_sparse.py
import os
import resource
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from scipy import sparse as sp

# import the module you wrote above
from piano.utils import triton_sparse
from piano.utils.triton_sparse import SparseTritonMatrix 


CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="GPU with CUDA required for Triton kernels"
)


# ---------------------------------------------------------------------------
# Helper: create a random CSR matrix with controllable shape / density
# ---------------------------------------------------------------------------
def make_random_csr(n_rows, n_cols, density=0.05, seed=0):
    rng = np.random.default_rng(seed)
    nnz = int(n_rows * n_cols * density)
    row = rng.integers(0, n_rows, size=nnz, dtype=np.int32)
    col = rng.integers(0, n_cols, size=nnz, dtype=np.int32)
    data = rng.random(size=nnz, dtype=np.float32)
    return sp.csr_matrix((data, (row, col)), shape=(n_rows, n_cols))


# ---------------------------------------------------------------------------
# Parametrised test cases
#
#   * small_cols → indices should be uint16  (≤ 65 535)
#   * large_cols → indices should be int32   (>  65 535, < 2³¹)
# ---------------------------------------------------------------------------
CASES = [
    pytest.param(
        dict(n_rows=256, n_cols=50_000, density=0.03),
        torch.uint16 if hasattr(torch, "uint16") else torch.int16,
        id="uint16_indices",
    ),
    pytest.param(
        dict(n_rows=256, n_cols=70_000, density=0.03),
        torch.int32,
        id="int32_indices",
    ),
]


@pytest.mark.parametrize("csr_kwargs, expected_idx_dtype", CASES)
def test_sparse_cuda_matrix_correctness(csr_kwargs, expected_idx_dtype):
    """Compare Triton result with reference SciPy for various index widths."""
    X = make_random_csr(**csr_kwargs)
    sm = SparseTritonMatrix(X)

    # ---- dtype assertions -------------------------------------------------
    assert sm.indices.dtype == expected_idx_dtype
    assert sm.indptr.dtype == torch.int32  # indptr stays i32 for these sizes

    # ---- numerical equality ----------------------------------------------
    rng = np.random.default_rng(42)
    for _ in range(5):  # five random batches
        rows = rng.integers(0, X.shape[0], size=64, dtype=np.int64)
        dense_gpu = sm[rows].cpu().numpy()
        dense_ref = X[rows, :].toarray()
        np.testing.assert_allclose(dense_gpu, dense_ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Output row offsets past 2^31 (SCRUM-116)
#
#   The kernel writes row m of the dense (M, ncols) output at offset m * ncols.
#   With 32770 rows of 65535 columns the last row starts at
#   32769 * 65535 = 2147516415 > 2^31 - 1, which overflowed in int32.
# ---------------------------------------------------------------------------
LARGE_NCOLS = 65535                      # widest matrix that still uses uint16 indices
LARGE_M = 2**31 // LARGE_NCOLS + 2       # 32770 rows: the last row starts past 2^31 - 1

# Free GPU memory the end-to-end test needs: the uint16 output takes LARGE_M * LARGE_NCOLS * 2 bytes = 4 GiB,
# plus 1 GiB of headroom for the child process's CUDA context and the Triton compilation
EXPAND_PAST_INT32_FREE_BYTES = 5 * 2**30

# Repository root, so the subprocesses import this checkout of piano
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@triton.jit
def _store_output_row_offsets(out_ptr, ncols: tl.constexpr):
    # One program per row: store the offset the expand kernel uses for row m
    m = tl.program_id(0)
    tl.store(out_ptr + m, triton_sparse._output_row_offset(m, ncols).to(tl.int64))


def test_output_row_offset_does_not_overflow_int32():
    """The output row offset helper returns m * ncols exactly, also past 2^31 (needs almost no GPU memory)."""
    offsets = torch.zeros(LARGE_M, dtype=torch.int64, device="cuda")
    _store_output_row_offsets[(LARGE_M,)](offsets, LARGE_NCOLS)

    # An int32 offset wraps around to -2147450881 for the last row
    expected = np.arange(LARGE_M, dtype=np.int64) * LARGE_NCOLS
    assert offsets[-1].item() == (LARGE_M - 1) * LARGE_NCOLS == 2147516415
    np.testing.assert_array_equal(offsets.cpu().numpy(), expected)


# Child process for the end-to-end test: it exits non-zero on a wrong result or a CUDA error
EXPAND_PAST_INT32_SCRIPT = textwrap.dedent('''
    import sys

    import numpy as np
    import torch
    from scipy import sparse as sp

    from piano.utils.triton_sparse import SparseTritonMatrix, expand_rows

    indptr_dtype, ncols, n_rows = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

    # Two rows: row 0 is empty, row 1 has one count (7) in the last column
    X = sp.csr_matrix(
        (np.array([7], dtype=np.uint16), np.array([ncols - 1]), np.array([0, 0, 1])),
        shape=(2, ncols),
    )
    sm = SparseTritonMatrix(X)
    assert sm.indptr.dtype == torch.int32 and sm.indices.dtype == torch.uint16

    # Only the last output row is non-empty, so the only write lands past offset 2^31
    rows = torch.zeros(n_rows, dtype=torch.int64, device="cuda")
    rows[-1] = 1
    if indptr_dtype == "int32":
        out = sm[rows]
    else:
        # SparseTritonMatrix only picks int64 indptr past 2^31 counts: cast it to cover that kernel path
        out = expand_rows(sm.indptr.to(torch.int64), sm.indices, sm.data, rows, ncols)
    torch.cuda.synchronize()

    assert out.shape == (n_rows, ncols) and out.dtype == torch.uint16
    assert out[-1, -1].item() == 7, out[-1, -1].item()
    assert not out[-1, :-1].view(torch.uint8).any().item(), "unexpected values in the last row"
    assert not out[:-1].view(torch.uint8).any().item(), "unexpected values in the other rows"
''')


# Opt-in (large_gpu, left out of default runs by pyproject.toml): run it with `pytest -m large_gpu` in its own MIG job
# whenever triton_sparse.py or the SparseGPU code changes. Selected on a GPU without enough free memory, it fails
# rather than skips, so a run that was meant to cover it never passes silently without it
@pytest.mark.large_gpu
@pytest.mark.parametrize("indptr_dtype", ["int32", "int64"], ids=["int32_indptr", "int64_indptr"])
def test_expand_rows_past_int32_output_offset(indptr_dtype):
    """expand_rows returns the right rows when the output row offset passes 2^31 (about 4 GiB of GPU memory)."""
    # Release this process's cached GPU memory from earlier tests, so the child process can use it
    torch.cuda.empty_cache()
    free_bytes = torch.cuda.mem_get_info()[0]
    if free_bytes < EXPAND_PAST_INT32_FREE_BYTES:
        pytest.fail(
            f"needs {EXPAND_PAST_INT32_FREE_BYTES / 2**30:.0f} GiB of free GPU memory, "
            f"{free_bytes / 2**30:.1f} GiB free: run it alone on a GPU with more free memory (e.g. a MIG 1g.10gb slice)"
        )

    # Run in a child process without core dumps: an out-of-bounds write is a CUDA illegal memory access,
    # which leaves the CUDA context unusable and would break every later GPU test in this session
    result = subprocess.run(
        [sys.executable, "-c", EXPAND_PAST_INT32_SCRIPT, indptr_dtype, str(LARGE_NCOLS), str(LARGE_M)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")]))},
        capture_output=True, text=True, timeout=600,
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)),
    )
    assert result.returncode == 0, result.stderr[-3000:]
