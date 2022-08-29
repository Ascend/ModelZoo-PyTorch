# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from collections.abc import Mapping, Sequence


def npu(o, id=0):
    """
    Applies npu recursively to modules and tensors.
    """
    if isinstance(o, Mapping):
        return type(o)((k, npu(v,id=id)) for (k, v) in o.items())
    if isinstance(o, Sequence):
        return type(o)(npu(v) for v in o)
    if hasattr(o, 'npu'):
        return o.npu(id)
    return o
