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


import argparse
import torch.onnx
from easydict import EasyDict as edict
from SFA3D.sfa.models.model_utils import create_model  # provided by SFA3D official github code repository


def parse_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='./fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--onnx_path', type=str, default='SFA3D.onnx', metavar='PATH',
                        help='the path to save onnx model')
    parser.add_argument('--k', type=int, default=50, help='the number of top k')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size (default:4)')
    parser.add_argument('--peak_tresh', type=float, default=0.2)

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.dowm_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4
    return configs


def convert2onnx():
    configs = parse_configs()
    model = create_model(configs)
    print('\n' + '-*=' * 30 + '\n')

    model.load_state_dict(torch.load(configs.pretrained_path, map_location=torch.device("cpu")))
    model.eval()
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    input_tensor = torch.randn(1, 3, 608, 608)
    torch.onnx.export(
        model,
        input_tensor,
        configs.onnx_path,
        input_names=["inputs"],
        output_names=["output1", "output2", "output3", "output4", "output5"],
        verbose=True,
        opset_version=11,
        dynamic_axes={
            # dynamic batchsize
            "inputs": {0: "-1"},
            "output1": {0: "-1"}, "output2": {0: "-1"}, "output3": {0: "-1"}, "output4": {0: "-1"}, "output5": {0: "-1"}
        }
    )


if __name__ == '__main__':
    convert2onnx()