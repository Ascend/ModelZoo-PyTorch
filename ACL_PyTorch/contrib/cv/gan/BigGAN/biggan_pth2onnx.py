# Copyright 2021 Huawei Technologies Co., Ltd
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

import onnx
import torch
import argparse

from BigGAN import Generator
from collections import OrderedDict


def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if (k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]
        new_state_dict[name] = v
    return new_state_dict


def pth2onnx(input_file, output_file):
    checkpoint = torch.load(input_file, map_location=torch.device('cpu'))
    checkpoint = proc_nodes_module(checkpoint)

    model = Generator(**{'G_lr':1e-4, 'SN_eps':1e-6, 'adam_eps':1e-6,
                         'G_ch':96, 'shared_dim':128,
                         'skip_init':True, 'no_optim': True,
                         'hier':True, 'dim_z':120})
    model.load_state_dict(checkpoint)
    model.eval()

    input_names = ['noise', 'label']
    output_names = ['image']
    dynamic_axes = {'noise': {0: '-1'}, 'label': {0: '-1'}, 'image': {0: '-1'}}

    dummy_z = torch.randn((1, 1, 20))
    dummy_y = torch.randn((1, 5, 148))

    torch.onnx.export(model, (dummy_z, dummy_y), output_file, dynamic_axes=dynamic_axes,
                      verbose=True, input_names=input_names, output_names=output_names, opset_version=11)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="./G_ema.pth")
    parser.add_argument('--target', type=str, default="./biggan.onnx")
    args = parser.parse_args()

    pth2onnx(args.source, args.target)
