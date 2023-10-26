# Copyright 2020 Huawei Technologies Co., Ltd
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
import densetorch as dt

from RefineNet_pytorch.models.resnet import rf101

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='model/RefineNet.pth.tar')
    parser.add_argument('--output-file', type=str, default='model/RefineNet.onnx')
    args = parser.parse_args()

    num_classes = 21
    model = rf101(num_classes, imagenet=False, pretrained=False).cpu()

    # Checkpoint
    model_state_dict = torch.load(args.input_file, map_location='cpu').get('model')
    dt.misc.load_state_dict(model, model_state_dict)

    model.eval()
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    torch.onnx.export(model, torch.randn(1, 3, 500, 500), args.output_file, 
                    input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes, opset_version=11, verbose=True)