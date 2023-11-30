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


import argparse
import torch
import torch.onnx
import torchvision.models as models


def parse_args():
    parser = argparse.ArgumentParser(description='MobileNet_V2 Export Model.')
    parser.add_argument('--input_path', type=str, default='./mobilenet_v2-b0353104.pth',
                        help='Original TorchScript model path')
    parser.add_argument('--output_path', type=str, default='./mobilenet_v2.ts',
                        help='Target TorchScript model path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    return parser.parse_args()


def convert():
    model = models.mobilenet_v2(pretrained=False)
    pthfile = args.input_path
    mobilenet_v2 = torch.load(pthfile, map_location='cpu')
    model.load_state_dict(mobilenet_v2)

    dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size)

    model.eval()
    ts_model = torch.jit.trace(model, dummy_input)
    ts_model.save(args.output_path)


if __name__ == "__main__":
    args = parse_args()
    convert()
