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
from unittest import TestCase
from unittest.mock import patch

import mmcv

from mmcls.utils import auto_select_device


class TestAutoSelectDevice(TestCase):

    @patch.object(mmcv, '__version__', '1.6.0')
    @patch('mmcv.device.get_device', create=True)
    def test_mmcv(self, mock):
        auto_select_device()
        mock.assert_called_once()

    @patch.object(mmcv, '__version__', '1.5.0')
    @patch('torch.cuda.is_available', return_value=True)
    def test_cuda(self, mock):
        device = auto_select_device()
        self.assertEqual(device, 'cuda')

    @patch.object(mmcv, '__version__', '1.5.0')
    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu(self, mock):
        device = auto_select_device()
        self.assertEqual(device, 'cpu')
