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
import pickle

import pytest
import torch
import torch.storage

from piano import Composer

# The text every refusal of Composer.load ends with
TRUST_MESSAGE = 'Only load files you trust'


# ---------------------------------------------------------------------------
# Composer.load refuses pickles that hold anything a Composer saved by Composer.save does not hold
#
# Each crafted pickle below would run a harmless command if it were unpickled: it creates the file 'marker' in tmp_path,
# a new, empty directory pytest makes for each test. The tests check that Composer.load raises pickle.UnpicklingError and
# that the marker does not exist, so the command did not run. On master the commands run.
# ---------------------------------------------------------------------------
class Reduce:
    """An object that pickles as a call of func with args, as a crafted pickle would hold."""
    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def __reduce__(self):
        return self.func, self.args


def write_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path


def touch_command(marker):
    return f'touch {marker}'


def test_load_refuses_code_in_pickled_tensor_bytes(tmp_path):
    # torch.storage._load_from_bytes, which a Composer pickle uses for its tensors, unpickles the bytes it is given with
    # torch.load. Here the bytes are a pickle that runs the command
    marker = tmp_path / 'marker'
    payload = pickle.dumps(Reduce(os.system, touch_command(marker)))
    path = write_pickle(Reduce(torch.storage._load_from_bytes, payload), tmp_path / 'pianist.pkl')

    # Previously the command ran under torch.load(weights_only=False), which then failed with RuntimeError
    # 'Invalid magic number; corrupt file?'. piano's message replaces torch's, which suggests weights_only=False
    with pytest.raises(pickle.UnpicklingError, match=TRUST_MESSAGE) as excinfo:
        Composer.load(path, device='cpu')
    assert not marker.exists()
    # torch's own error is kept as the cause ('from err'), and its advice to set weights_only=False is not shown
    assert isinstance(excinfo.value.__cause__, pickle.UnpicklingError)
    assert 'weights_only=False' not in str(excinfo.value)
