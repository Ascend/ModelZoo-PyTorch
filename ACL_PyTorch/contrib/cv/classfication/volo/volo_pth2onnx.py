# Copyright 2023 Huawei Technologies Co., Ltd
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
import torch.onnx
from timm.models import create_model, load_checkpoint
import os
from volo import *
import argparse

def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = create_model(
        'volo_d1',
        pretrained=False,
        num_classes=None,
        in_chans=3,
        global_pool=None,
        scriptable=False,
        img_size=224)
    load_checkpoint(model, checkpoint, False, strict=False)
    model.eval()

    try:
        torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names,
                          opset_version=11)
    except torch.onnx.utils.ONNXCheckerError:
        print("ONNX graph is invalid but the ONNX file is still generated.")
    print("Exporting .pth model to onnx model has been successful!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch to onnx')
    parser.add_argument('--src', type=str, default='./d1_224_84.2.pth.tar',
                        help='weights of pytorch dir')
    parser.add_argument('--des', type=str, default='./volo_d1_224_Col2im.onnx',
                        help='weights of onnx dir')
    parser.add_argument('--batchsize', type=int, default='1',
                        help='batchsize.')
    args = parser.parse_args()
    checkpoint = args.src
    onnx_path = args.des
    bs = args.batchsize
    input = torch.randn(bs, 3, 224, 224)
    pth_to_onnx(input, checkpoint, onnx_path)



