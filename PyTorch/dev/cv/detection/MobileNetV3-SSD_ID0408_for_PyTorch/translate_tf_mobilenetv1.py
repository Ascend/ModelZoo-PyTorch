#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
import torch
import sys

from vision.nn.mobilenet import MobileNetV1
from extract_tf_weights import read_weights


def fill_weights_torch_model(weights, state_dict):
    for name in state_dict:
        if name == 'classifier.weight':
            weight = weights['MobilenetV1/Logits/Conv2d_1c_1x1/weights']
            weight = torch.tensor(weight, dtype=torch.float32).permute(3, 2, 0, 1)
            assert state_dict[name].size() == weight.size()
            state_dict[name] = weight
        elif name == 'classifier.bias':
            bias = weights['MobilenetV1/Logits/Conv2d_1c_1x1/biases']
            bias = torch.tensor(bias, dtype=torch.float32)
            assert state_dict[name].size() == bias.size()
            state_dict[name] = bias
        elif name.endswith('BatchNorm.weight'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('BatchNorm/weight', 'BatchNorm/gamma')
            weight = torch.tensor(weights[key], dtype=torch.float32)
            assert weight.size() == state_dict[name].size()
            state_dict[name] = weight
        elif name.endswith('BatchNorm.bias'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('BatchNorm/bias', 'BatchNorm/beta')
            bias = torch.tensor(weights[key], dtype=torch.float32)
            assert bias.size() == state_dict[name].size()
            state_dict[name] = bias
        elif name.endswith('running_mean'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('running_mean', 'moving_mean')
            running_mean = torch.tensor(weights[key], dtype=torch.float32)
            assert running_mean.size() == state_dict[name].size()
            state_dict[name] = running_mean
        elif name.endswith('running_var'):
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('running_var', 'moving_variance')
            running_var = torch.tensor(weights[key], dtype=torch.float32)
            assert running_var.size() == state_dict[name].size()
            state_dict[name] = running_var
        elif name.endswith('depthwise.weight'):
            key = name.replace("features", "MobilenetV1").replace(".", "/")
            key = key.replace('depthwise/weight', 'depthwise/depthwise_weights')
            weight = torch.tensor(weights[key], dtype=torch.float32).permute(2, 3, 0, 1)
            assert weight.size() == state_dict[name].size()
            state_dict[name] = weight
        else:
            key = name.replace("features", "MobilenetV1").replace(".", "/").replace('weight', 'weights')
            weight = torch.tensor(weights[key], dtype=torch.float32).permute(3, 2, 0, 1)
            assert weight.size() == state_dict[name].size()
            state_dict[name] = weight


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python translate_tf_modelnetv1.py <tf_model.pb> <pytorch_weights.pth>")
    tf_model = sys.argv[1]
    torch_weights_path = sys.argv[2]
    print("Extract weights from tf model.")
    weights = read_weights(tf_model)

    net = MobileNetV1(1001)
    states = net.state_dict()
    print("Translate tf weights.")
    fill_weights_torch_model(weights, states)
    torch.save(states, torch_weights_path)