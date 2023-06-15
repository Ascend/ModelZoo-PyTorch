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

import argparse
from collections import OrderedDict
import torch
import torch.onnx
from experiment import Structure, Experiment
from concern.config import Configurable, Config

def proc_nodes_modile(checkpoint_inp):
    new_state_dict = OrderedDict()
    for k, v in checkpoint_inp.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def pth2onnx(model_inp):
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 736, 1280)

    torch.onnx.export(model_inp, 
                    dummy_input, 
                    "dbnet.onnx", 
                    input_names=input_names, 
                    enable_onnx_checker=False, 
                    dynamic_axes=dynamic_axes, 
                    output_names=output_names, 
                    opset_version=11, 
                    verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='db pth2onnx')
    parser.add_argument('exp', type=str)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)
    model = experiment.structure.builder.build(torch.device('cpu'))
    checkpoint = torch.load(args['resume'], map_location=torch.device('cpu'))
    checkpoint = proc_nodes_modile(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    pth2onnx(model)
