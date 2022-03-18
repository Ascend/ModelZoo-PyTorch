# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CommandLine:
    pytest tests/test_nms.py
"""
import numpy as np
import torch

from mmdet.ops.nms.nms_wrapper import nms


def test_nms_device_and_dtypes_cpu():
    """
    CommandLine:
        xdoctest -m tests/test_nms.py test_nms_device_and_dtypes_cpu
    """
    iou_thr = 0.7
    base_dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
                          [49.3, 32.9, 51.0, 35.3, 0.9],
                          [35.3, 11.5, 39.9, 14.5, 0.4],
                          [35.2, 11.7, 39.7, 15.7, 0.3]])

    # CPU can handle float32 and float64
    dets = base_dets.astype(np.float32)
    supressed, inds = nms(dets, iou_thr)
    assert dets.dtype == supressed.dtype
    assert len(inds) == len(supressed) == 3

    dets = torch.FloatTensor(base_dets)
    surpressed, inds = nms(dets, iou_thr)
    assert dets.dtype == surpressed.dtype
    assert len(inds) == len(surpressed) == 3

    dets = base_dets.astype(np.float64)
    supressed, inds = nms(dets, iou_thr)
    assert dets.dtype == supressed.dtype
    assert len(inds) == len(supressed) == 3

    dets = torch.DoubleTensor(base_dets)
    surpressed, inds = nms(dets, iou_thr)
    assert dets.dtype == surpressed.dtype
    assert len(inds) == len(surpressed) == 3


def test_nms_device_and_dtypes_gpu():
    """
    CommandLine:
        xdoctest -m tests/test_nms.py test_nms_device_and_dtypes_gpu
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('test requires GPU and torch+cuda')

    iou_thr = 0.7
    base_dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
                          [49.3, 32.9, 51.0, 35.3, 0.9],
                          [35.3, 11.5, 39.9, 14.5, 0.4],
                          [35.2, 11.7, 39.7, 15.7, 0.3]])

    for device_id in range(torch.cuda.device_count()):
        print('Run NMS on device_id = {!r}'.format(device_id))
        # GPU can handle float32 but not float64
        dets = base_dets.astype(np.float32)
        supressed, inds = nms(dets, iou_thr, device_id)
        assert dets.dtype == supressed.dtype
        assert len(inds) == len(supressed) == 3

        dets = torch.FloatTensor(base_dets).to(device_id)
        surpressed, inds = nms(dets, iou_thr)
        assert dets.dtype == surpressed.dtype
        assert len(inds) == len(surpressed) == 3
