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
import argparse
import json
from importlib import import_module
from pathlib import Path

import torch
import torch.nn as nn
import onnx

ROOT = './'
if ROOT not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH

def convert(args):
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]

    # define model architecture
    model = get_model(model_config, "cpu")
    model.load_state_dict(torch.load(config["model_path"], map_location="cpu"))
    model.eval()

    input_names = ["input"]
    output_names = ["output"]
    dummy_input = torch.randn(args.batch_size, 64600)

    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    torch.onnx.export(model, dummy_input, args.onnx_model,
                      input_names=input_names, output_names=output_names,
                      opset_version=11,
                      dynamic_axes=None,
                      export_params=True,
                      verbose=False,
                      do_constant_folding=True)
    
    # Checks
    model_onnx = onnx.load(args.onnx_model)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Simplify
    try:
        import onnxsim
        print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
        model_onnx, check = onnxsim.simplify(
            model_onnx,
            dynamic_input_shape=None,
            input_shapes=None)
        assert check, 'assert check failed'
        onnx.save(model_onnx, args.onnx_model)
    except Exception as e:
        print(f'simplifier failure: {e}')


def get_model(model_config, device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",type=str, default='./config/AASIST-L.conf', help="configuration file")
    parser.add_argument('--onnx-model', default='aasist_bs1.onnx', type=str, help='Path to onnx model')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    args = parser.parse_args()
    convert(args)
