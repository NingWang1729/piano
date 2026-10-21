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

import os

# Deterministic GPU training needs CUBLAS_WORKSPACE_CONFIG, which torch reads only once, at the first GPU matrix product in
# the process. A Composer with deterministic=True (the default) sets it, which is enough in a fresh process such as a
# user's script, but in this test session earlier tests may already have used the GPU. Set it for the whole session, before
# any test runs; tests/test_determinism.py checks the Composer's own setting in a separate process
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
