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
from collections import OrderedDict
import torch

sys.path.append(r"PyTorch-GAN/implementations/dcgan")
from dcgan import Generator


def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def pth2onnx(input_file, output_file):
    shape = (1, 100, 1, 1)
    img_size = 32
    latent_dim = 100
    channels = 1
    model = Generator(img_size, latent_dim, channels)
    checkpoint = torch.load(input_file, map_location='cpu')['G']
    checkpoint = proc_nodes_module(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    input_names = ["noise"]
    output_names = ["image"]
    dynamic_axes = {'noise': {0: '-1'}, 'image': {0: '-1'}}
    dummy_input = torch.randn(shape)
    torch.onnx.export(model, dummy_input, output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=True)


if __name__ == "__main__":
    # sys.argv[1]:输入的pth模型的路径
    # sys.argv[2]:期望输出的onnx模型的路径
    pth2onnx(sys.argv[1], sys.argv[2])
