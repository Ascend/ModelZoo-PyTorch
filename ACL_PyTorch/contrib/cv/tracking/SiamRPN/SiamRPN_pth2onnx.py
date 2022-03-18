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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import torch

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.utils.model_load import load_pretrain

parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
                    help='datasets')
parser.add_argument('--config', default='', type=str,
                    help='config file')
parser.add_argument('--snapshot', default='', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
                    help='eval one special video')
parser.add_argument('--vis', action='store_true',
                    help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)


def main():
    cfg.merge_from_file(cfg.CONFIG_PATH)

    model = ModelBuilder()

    model = load_pretrain(model, cfg.MODEL_PATH)

    a = torch.ones([1, 3, 127, 127])
    b = torch.ones([1, 3, 255, 255])
    c = torch.ones([1, 5, 25, 25], dtype=torch.long)
    d = torch.ones([1, 4, 5, 25, 25])
    e = torch.ones([1, 5, 25, 25])
    data = {'template': a,
            'search': b,
            'label_cls': c,
            'label_loc': d,
            'label_loc_weight': e
            }

    def pth2onnx(output_file):
        model.eval()
        input_names = ['template', 'search', 'label_cls', 'label_loc', 'label_loc_weight']
        output_names = ['cls', 'loc']
        dynamic_axes = {'template': {0: '1'}, 'search': {0: '1'}, 'label_cls': {0: '1'}, 'label_loc': {0: '1'},
                        'label_loc_weight': {0: '1'}, 'cls': {0: '1'}, 'loc': {0: '1'}}
        dummy_input = data

        torch.onnx.export(model, dummy_input, output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                          output_names=output_names, opset_version=11, verbose=False)

    pth2onnx('SiamRPN.onnx')


if __name__ == '__main__':
    main()
