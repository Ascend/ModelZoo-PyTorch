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
import mmcv
import torch
from mmcv.utils import digit_version


def auto_select_device() -> str:
    mmcv_version = digit_version(mmcv.__version__)
    if mmcv_version >= digit_version('1.6.0'):
        from mmcv.device import get_device
        return get_device()
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
