#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
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

import numpy as np
import torch

from concern.config import Configurable


class TransformData(Configurable):
    '''
    this transformation is inplcae, which means that the input
        will be modified.
    '''
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def __call__(self, data_dict, *args, **kwargs):
        image = data_dict['image'].transpose(2, 0, 1)
        image = image / 255.0
        image = (image - self.mean[:, None, None]) / self.std[:, None, None]
        data_dict['image'] = image
        return data_dict
