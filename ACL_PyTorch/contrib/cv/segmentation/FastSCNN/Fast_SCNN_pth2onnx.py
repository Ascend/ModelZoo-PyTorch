# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import torch.onnx
from collections import OrderedDict
import sys
sys.path.append('./SegmenTron')
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.config import cfg
import ssl

def pth2onnx():
    model = get_segmentation_model()
    checkpoint = torch.load(args.pth_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    input_names = ["image"]
    output_names = ["output1"]
    dynamic_axes = {'image': {0: '-1'}, 'output1': {0: '-1'}}
    dummy_input1 = torch.randn(args.batch_size, 3, 1024, 2048)
    output_file1 = args.onnx_name + '.onnx'
    torch.onnx.export(model, dummy_input1, output_file1, input_names = input_names,dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11, verbose=True)
    print(args.onnx_name,"batchsize",args.batch_size," onnx has transformed successfully")
    print('onnx export done.')
    

if __name__ == "__main__":
    args = parse_args()
    args.config_file = 'SegmenTron/configs/cityscapes_fast_scnn.yaml'
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    pth2onnx()

