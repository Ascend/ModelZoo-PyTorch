#     Copyright 2021 Huawei
#     Copyright 2021 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#

import torch

from mmseg.models.decode_heads import FCNHead, OCRHead
from .utils import to_cuda


def test_ocr_head():

    inputs = [torch.randn(1, 32, 45, 45)]
    ocr_head = OCRHead(
        in_channels=32, channels=16, num_classes=19, ocr_channels=8)
    fcn_head = FCNHead(in_channels=32, channels=16, num_classes=19)
    if torch.cuda.is_available():
        head, inputs = to_cuda(ocr_head, inputs)
        head, inputs = to_cuda(fcn_head, inputs)
    prev_output = fcn_head(inputs)
    output = ocr_head(inputs, prev_output)
    assert output.shape == (1, ocr_head.num_classes, 45, 45)
