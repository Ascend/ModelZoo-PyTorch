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

import os  
import sys  
sys.path.append('./exps/mspn.2xstg.coco/')        
import torch
from config import cfg
from network import MSPN
import torch.onnx


def main():

    model = MSPN(cfg)

    model_file = os.path.join("mspn_2xstg_coco.pth")
    if os.path.exists(model_file):
        print('MSPN loaded')
        state_dict = torch.load(
                model_file, map_location=lambda storage, loc: storage)
        state_dict = state_dict['model']
        model.load_state_dict(state_dict)
    model.eval()

    dummy_input= torch.randn(32, 3, 256, 192)
    dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}
    export_onnx_file = "MSPN.onnx"
    torch.onnx.export(model,               # model being run
                    dummy_input,                         # model input (or a tuple for multiple inputs)
                    export_onnx_file,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes=dynamic_axes,    # variable lenght axes
                    )


if __name__ == '__main__':
    main()
