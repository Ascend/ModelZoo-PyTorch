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

from __future__ import print_function

import sys
sys.path.insert(0, './AlignedReID-Re-Production-Pytorch')

import torch
from aligned_reid.model.Model import Model


class Config(object):
    def __init__(self):
        ###############
        # ReID Model  #
        ###############
        self.local_conv_out_channels = 128


def main(pth_path, out_name, bs):
    cfg = Config()

    ##############
    # OutputOnnx #
    ##############
    model = Model(local_conv_out_channels=cfg.local_conv_out_channels,
                  num_classes=751)

    if pth_path != '':
        model.load_state_dict(torch.load(pth_path, map_location='cpu')['state_dicts'][0])
    model.eval()

    input_names = ["image"]
    output_names = ["global_feat", "local_feat", "logits"]

    dynamic_axes = {'image': {0: bs}, 'class': {0: bs}}
    dummy_input = torch.randn(int(bs), 3, 256, 128)

    torch.onnx.export(
        model, dummy_input, out_name, input_names=input_names, output_names=output_names, verbose=True,
        opset_version=11, dynamic_axes=dynamic_axes)


if __name__ == '__main__':
    pth_path = sys.argv[1]
    out_name = sys.argv[2]
    bs = sys.argv[3]
    main(pth_path, out_name, bs)
