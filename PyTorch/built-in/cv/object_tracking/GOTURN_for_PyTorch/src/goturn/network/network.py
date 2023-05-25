
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

    
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
if torch.__version__ >= "1.8":
    import torch_npu

from loguru import logger
from torchsummary import summary

try:
    from goturn.network.caffenet import CaffeNet
except ImportError:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class FastGelu(nn.Module):
    def forward(self, input):
        return torch_npu.fast_gelu(input)

#  todo dropout -> NpuDropoutv2
class DroupoutV2(nn.Module):
    def __init__(self, p=0.5, inplace=False, max_seed=2 ** 10 - 1):
        super(DroupoutV2, self).__init__()
        self.p = p
        self.seed = torch.from_numpy(np.random.uniform(1, max_seed, size=(32 * 1024 * 12,)).astype(np.float32))
        self.checked = False

    def check_self(self, x):
        """Check device equipment between tensors.
        """
        if self.seed.device == x.device:
            self.checked = True
            return

        self.seed = self.seed.to(x.device)

    def forward(self, x):
        if not self.training:
            return x

        if not self.checked:
            self.check_self(x)
        if torch.__version__ > "1.8":
            x = nn.functional.dropout(x, p=self.p)
        else:
            x, mask, _ = torch.npu_dropoutV2(x, self.seed, p=self.p)
        return x


class GoturnNetwork(nn.Module):

    """AlexNet based network for training goturn tracker"""

    def __init__(self, pretrained_model=None,
                 init_fc=None, num_output=4):
        """ """
        super(GoturnNetwork, self).__init__()

        self._net_1 = CaffeNet(pretrained_model_path=pretrained_model)
        self._net_2 = CaffeNet(pretrained_model_path=pretrained_model)
        dropout_ratio = 0.5
        self._classifier = nn.Sequential(nn.Linear(256 * 6 * 6 * 2, 4096),
                                         nn.ReLU(inplace=True),
                                         DroupoutV2(dropout_ratio),
                                         nn.Linear(4096, 4096),
                                         nn.ReLU(inplace=True),
                                         DroupoutV2(dropout_ratio),
                                         nn.Linear(4096, 4096),
                                         nn.ReLU(inplace=True),
                                         DroupoutV2(dropout_ratio),
                                         nn.Linear(4096, num_output))

        self._num_output = num_output
        if init_fc:
            logger.info('Using caffe fc weights')
            self._init_fc = init_fc
            self._caffe_fc_init()
        else:
            logger.info('Not using caffe fc weights/ manually initialized')
            self.__init_weights()

    def forward(self, x1, x2):
        """Foward pass
        @x: input
        """
        self._net_1 = self._net_1.to(x1.device)
        x1 = self._net_1(x1)
        # x1 = x1.view(x1.size(0), 256 * 6 * 6)
        x1 = x1.reshape(x1.size(0), 256 * 6 * 6).contiguous()

        self._net_2 = self._net_2.to(x2.device)
        x2 = self._net_2(x2)
        # x2 = x2.view(x2.size(0), 256 * 6 * 6)
        x2 = x2.reshape(x2.size(0), 256 * 6 * 6).contiguous()

        x = torch.cat((x1, x2), 1)
        self._classifier = self._classifier.to(x.device)
        x = self._classifier(x)

        return x

    def __init_weights(self):
        """Initialize the extra layers """
        for m in self._classifier.modules():
            if isinstance(m, nn.Linear):
                if self._num_output == m.out_features:
                    init.normal_(m.weight.data, mean=0.0, std=0.01)
                    init.zeros_(m.bias.data)
                else:
                    init.normal_(m.weight.data, mean=0.0, std=0.005)
                    init.ones_(m.bias.data)

    def _caffe_fc_init(self):
        """Init from caffe normal_
        """
        wb = np.load(self._init_fc, allow_pickle=True).item()

        layer_num = 0
        with torch.no_grad():
            for layer in self._classifier.modules():
                if isinstance(layer, nn.Linear):
                    layer_num = layer_num + 1
                    key_w = 'fc{}_w'.format(layer_num)
                    key_b = 'fc{}_b'.format(layer_num)
                    w, b = wb[key_w], wb[key_b]
                    w = np.reshape(w, (w.shape[1], w.shape[0]))
                    b = np.squeeze(np.reshape(b, (b.shape[1],
                                                  b.shape[0])))
                    layer.weight.copy_(torch.from_numpy(w).float())
                    layer.bias.copy_(torch.from_numpy(b).float())


if __name__ == "__main__":
    net = GoturnNetwork().cuda()
    summary(net, input_size=[(3, 227, 227), (3, 227, 227)])
