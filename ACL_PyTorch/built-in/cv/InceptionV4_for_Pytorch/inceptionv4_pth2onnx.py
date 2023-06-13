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


import sys
import ssl

import os
from configparser import ConfigParser
import torch
import torch.onnx
import torch.utils.model_zoo as model_zoo


from pretrainedmodels.models.inceptionv4 import InceptionV4
sys.path.append(r"./pretrained-models.pytorch")
config = ConfigParser()
config.read(filenames='url.ini',encoding = 'UTF-8')
value = config.get(section="DEFAULT", option="pth_url")


url = str(value)
pretrained_settings = {
    'inceptionv4': {
        'imagenet': {
            'url': url,
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': url,
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}


def inceptionv4(num_classes=1000, pretrained='imagenet', localpath=None):
    if pretrained:
        settings = pretrained_settings['inceptionv4'][pretrained]
        assert num_classes is settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], 
                                                         num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionV4(num_classes=1001)
        if localpath is None:
            model.load_state_dict(model_zoo.load_url(settings['url']))
        else:
            checkpoint = torch.load(localpath)
            model.load_state_dict(checkpoint)

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    return model

def convert(checkpoint=None, output_file=None,):
    # https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py
    model = inceptionv4(num_classes=1001, pretrained='imagenet+background', 
                        localpath=checkpoint)
    model.eval()
    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 299, 299)

    torch.onnx.export(model, dummy_input, output_file, 
                      input_names = input_names,
                      output_names = output_names,
                      dynamic_axes = dynamic_axes, 
                      opset_version=11)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='InceptionResNetV2 Pytorch model convert to ONNX model')
    parser.add_argument('--ckpt', default=None, help='input checkpoint file path')
    parser.add_argument('--onnx', default='out.onnx', help='output onnx file path')
    args = parser.parse_args()

    ssl._create_default_https_context = ssl._create_unverified_context
    convert(args.ckpt, args.onnx)
