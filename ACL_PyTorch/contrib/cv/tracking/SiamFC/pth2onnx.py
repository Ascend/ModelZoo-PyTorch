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

import sys
import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F


response_scale = 1e-3


class SiameseAlexNet(nn.Module):
    def __init__(self):
        super(SiameseAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)
        )
        self.corr_bias = nn.Parameter(torch.zeros(1))
        self.exemplar = None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        exemplar, instance = x  # x = ( exemplar, instance )
        # train
        if exemplar is not None and instance is not None:  #
            batch_size = exemplar.shape[0]  #
            exemplar = self.features(exemplar)  # batch, 256, 6, 6
            instance = self.features(instance)  # batch, 256, 20, 20
            N, C, H, W = instance.shape
            instance = instance.view(1, -1, H, W)
            score = F.conv2d(instance, exemplar, groups=N) * response_scale + self.corr_bias
            return score.transpose(0, 1)
        # test(first frame)
        elif exemplar is not None and instance is None:
            self.exemplar = self.features(exemplar)  # 1, 256, 6, 6
            self.exemplar = torch.cat([self.exemplar for _ in range(3)], dim=0)  # 3, 256, 6, 6
            return self.exemplar
        # test(not first frame)
        else:
            # inference used we don't need to scale the response or add bias
            _, _, H, W = instance.shape
            instance = instance.reshape(3, 3, H, W)
            instance = self.features(instance)  # 3 scale
            N, C, H, W = instance.shape
            instance = instance.view(1, N*C, H, W)  # 1, NxC, H, W
            # score = F.conv2d(instance, self.exemplar, groups=N)
            # return score.transpose(0, 1)
            return instance


def exemplar_convert(input_file, output_file):
    model = SiameseAlexNet()
    model.load_state_dict(torch.load(input_file, map_location='cpu'))
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    input1 = torch.randn(1, 3, 127, 127)
    input2 = None
    dummy_input = [input1, input2]
    torch.onnx.export(model, dummy_input, output_file, input_names=input_names, output_names=output_names,
                      opset_version=11)


def search_convert(input_file, output_file):
    model = SiameseAlexNet()
    model.load_state_dict(torch.load(input_file, map_location='cpu'))
    model.eval()

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    input1 = None
    input2 = torch.randn(1, 9, 255, 255)
    dummy_input = [input1, input2]
    torch.onnx.export(model, dummy_input, output_file, input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file_exemplar = sys.argv[2]
    output_file_search = sys.argv[3]
    exemplar_convert(input_file, output_file_exemplar)
    search_convert(input_file, output_file_search)
