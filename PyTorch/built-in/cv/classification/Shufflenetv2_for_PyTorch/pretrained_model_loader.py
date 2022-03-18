# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import torch
import models

def load_state_dict(module, state_dict):
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    if not isinstance(state_dict, dict):
        print(state_dict)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    print('\n'.join(error_msgs))


if __name__ == '__main__':
    import urllib.request
    import os

    path = 'shufflenetv2_x1-5666bf0f80.pth'
    if not os.path.exists(path):
        url = models.shufflenetv2_wock_op_woct_8p.model_urls['shufflenetv2_x1.0']
        urllib.request.urlretrieve(url, path)
    state_dict = torch.load(path, map_location='cpu')
    print(state_dict.keys())

    model = models.shufflenet_v2_x1_0(num_classes=10)
    load_state_dict(model, state_dict)

