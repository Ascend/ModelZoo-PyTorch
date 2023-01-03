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
import torch
import argparse
import sys
sys.path.append(r"./BSN-boundary-sensitive-network.pytorch")

from models import PEM

parser = argparse.ArgumentParser(
    description='pem2onnx')
parser.add_argument(
        '--pth_path',
        default='./pem_best.pth.tar',
        help='pth path'
        ) 
parser.add_argument(
        '--onnx_path',
        default='./BSN_pem.onnx',
        help='onnx path'
        )
parser.add_argument(
        '--pem_feat_dim',
        type=int,
        default=32
        )
parser.add_argument(
        '--pem_hidden_dim',
        type=int,
        default=256
        )
parser.add_argument(
        '--pem_batch_size',
        type=int,
        default=16
        )
parser.add_argument(
        '--pem_u_ratio_m',
        type=float,
        default=1
        )
parser.add_argument(
        '--pem_u_ratio_l',
        type=float,
        default=2
        )


def pem_onnx(opt):
    opt = vars(opt)
    pth_path = opt['pth_path']
    onnx_path = opt['onnx_path']

    model = PEM(opt)
    checkpoint = torch.load(pth_path, map_location='cpu')
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    input_names=["video_feature"]
    output_names = ["output"]
    model.eval()
    dummy_input = torch.randn(1, 1000 ,32)
    dynamic_axes = {
        'video_feature': {0:'-1'},
        'output': {0:'-1'}
    }
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        verbose=True,
        opset_version=11,
        dynamic_axes=dynamic_axes
        )

if __name__ =="__main__":
    opt = parser.parse_args()
    pem_onnx(opt)