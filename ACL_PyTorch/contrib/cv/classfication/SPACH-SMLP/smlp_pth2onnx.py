# Copyright 2022 Huawei Technologies Co., Ltd
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
import torch
from timm.models import create_model
import models


def pytorch2onnx(model_name, pth_path, onnx_path, opset_version):

    device = torch.device('cpu')
    seed = 20200220
    torch.manual_seed(seed)

    model = create_model(
        model_name,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    model.to(device)
    checkpoint = torch.load(pth_path, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=False)

    dummy_input = torch.randn(1, 3, 224, 224)
    dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=True,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'pytorch pth convert to onnx', add_help=True)
    parser.add_argument('--model_name', default='smlpnet_tiny', type=str, metavar='MODEL',
                        help='Name of model to convert')
    parser.add_argument('--pth_path', default="smlp_t.pth", type=str,
                        help='path to checkpoint')
    parser.add_argument('--onnx_path', default="sMLPNet-T.onnx", type=str,
                        help='path to ONNX model')
    parser.add_argument('--opset_version', default=11, type=int,
                        help='opset version')
    args = parser.parse_args()

    pytorch2onnx(args.model_name, args.pth_path,
                 args.onnx_path, args.opset_version)
