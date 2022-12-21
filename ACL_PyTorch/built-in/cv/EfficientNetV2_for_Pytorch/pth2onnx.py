# Copyright 2021 Huawei Technologies Co., Ltd
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

import torch
import timm

model = timm.create_model('efficientnetv2_rw_t', pretrained=True)

dummy_input = torch.randn(1, 3, 288, 288)

torch.onnx.export(model,
                dummy_input,
                "efficientnetv2.onnx",
                input_names=['image'],
                output_names=['output'],
                dynamic_axes={
                    'image': {
                        0: 'batch_size'
                    },
                    'output': {
                        0: 'batch_size'
                    }
                }
)
