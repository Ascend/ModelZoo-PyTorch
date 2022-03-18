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

from facenet_pytorch import InceptionResnetV1
import torch
import argparse


def FaceNet_pth2onnx(opt):
    model = InceptionResnetV1(pretrained=opt.pretrain)
    # if opt.model != '':
    #    model.load_state_dict(torch.load(opt.model, map_location='cpu'))
    # else:
    #     print("Error network")
    #     return -1
    model.eval()
    input_names = ["image"]
    output_names = ["class"]
    output_file = opt.output_file
    if opt.output_file == '.':
        output_file = opt.output_file
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(16, 3, 160, 160)

    torch.onnx.export(model, dummy_input, output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, verbose=True, opset_version=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, default='vggface2', help='[casia-webface, vggface2]')
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--output_file', type=str, default='.', help='output path')
    arg = parser.parse_args()
    FaceNet_pth2onnx(arg)
