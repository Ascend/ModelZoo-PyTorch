# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities for fetching the Goodbooks-10K dataset [1]_.

References
----------

.. [1] https://github.com/zygmuntz/goodbooks-10k
"""

import h5py

import numpy as np

from spotlight.datasets import _transport
from spotlight.interactions import Interactions


def _get_dataset():

    path = _transport.get_data('https://github.com/zygmuntz/goodbooks-10k/'
                               'releases/download/v1.0/goodbooks-10k.hdf5',
                               'goodbooks',
                               'goodbooks.hdf5')

    with h5py.File(path, 'r') as data:
        return (data['ratings'][:, 0],
                data['ratings'][:, 1],
                data['ratings'][:, 2].astype(np.float32),
                np.arange(len(data['ratings']), dtype=np.int32))


def get_goodbooks_dataset():
    """
    Download and return the goodbooks-10K dataset [2]_.

    Returns
    -------

    Interactions: :class:`spotlight.interactions.Interactions`
        instance of the interactions class

    References
    ----------

    .. [2] https://github.com/zygmuntz/goodbooks-10k
    """

    return Interactions(*_get_dataset())
