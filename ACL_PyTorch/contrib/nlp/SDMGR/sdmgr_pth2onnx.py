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


import os
import sys
import glob
import random
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import torch
from mmcv.onnx import register_extra_symbolics
from mmocr.apis import init_detector

sys.path.append('mmocr/tools')
from deployment.pytorch2onnx import _convert_batchnorm


def create_input_data(prep_dir):
    prep_dir = Path(prep_dir)
    file_name = random.choice(os.listdir(prep_dir/'texts'))

    dummy_input = []
    for input_name in ['relations', 'texts', 'mask']:
        file_path = prep_dir/input_name/file_name
        data = torch.from_numpy(np.load(file_path))
        dummy_input.append(data)

    return tuple(dummy_input)


def pytorch2onnx(config_file, checkpoint_file, prep_dir, 
                 output_file, opset_version=12):
    device = torch.device(type='cpu')
    model = init_detector(config_file, checkpoint_file, device=device)
    if hasattr(model, 'module'):
        model = model.module
    model.to(torch.device('cpu')).eval()
    _convert_batchnorm(model)

    dummy_input = create_input_data(prep_dir)
    dynamic_axes = {
        'relations': {0: 'num_texts', 1: 'num_texts'},
        'texts': {0: 'num_texts', 1: 'num_chars'},
        'mask': {0: 'num_texts', 1: 'num_chars'},
        'nodes': {0: 'num_texts', 1: 'num_texts'},
        'edges': {0: 'num_edges'},
    }

    model.forward = model.forward_onnx
    model.bbox_head.forward = model.bbox_head.forward_onnx
    register_extra_symbolics(opset_version)
    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input,
            output_file,
            input_names=['relations', 'texts', 'mask'],
            output_names=['nodes', 'edges'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=False,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )
    print(f'Successfully exported ONNX model: {output_file}')
    

def main():
    parser = ArgumentParser('Convert MMOCR models from pytorch to ONNX')
    parser.add_argument('--config', type=str, help='config file.')
    parser.add_argument('--checkpoint', type=str, help='checkpint file.')
    parser.add_argument('--prep-dir', type=str, help='path to preprocessed data')
    parser.add_argument('--onnx', type=str, help='path to save onnx model.')
    parser.add_argument('--opset-version', type=int, default=12, 
                        help='ONNX opset version.')
    args = parser.parse_args()

    pytorch2onnx(args.config, args.checkpoint, args.prep_dir, 
                 args.onnx, opset_version=args.opset_version)


if __name__ == '__main__':
    main()
