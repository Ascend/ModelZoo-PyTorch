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
import torch
import sys
sys.path.append(r'./SRFlow/code')
from models import create_model
import options.options as option


parser = argparse.ArgumentParser(description='PyTorch Export ONNX')
parser.add_argument('--pth', default='', type=str, metavar='PATH',
                    help='path of pth file (default: none)')
parser.add_argument('--onnx', default='', type=str, metavar='PATH',
                    help='path of output (default: none)')
args = parser.parse_args()


def load_model(input_file):
    opt = option.parse(
        './SRFlow/code/confs/SRFlow_DF2K_8X.yml', is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)
    model_path = input_file
    md = torch.load(model_path)
    for name in md:
        if name.endswith('invconv.weight'):
            md[name] = torch.inverse(md[name].double()).float()
    inv_model_path = model_path[:-4] + '_inv.pth'
    torch.save(md, inv_model_path)
    model.load_network(load_path=inv_model_path, network=model.netG)
    return model, opt


def main():
    srflowmodel, _ = load_model(args.pth)
    model = srflowmodel.netG
    x = torch.randn(1, 3, 256, 256)
    model.eval()
    input_names = ["input_1"]
    output_names = ["output_1"]
    dynamic_axes = {'input_1': {0: '-1'}, 'output_1': {0: '-1'}}
    # Export the model
    torch.onnx.export(model, x, args.onnx, input_names=input_names,
                      output_names=output_names,dynamic_axes=dynamic_axes,
                      opset_version=11, verbose=True, 
                      do_constant_folding=True,export_params=True,
                      enable_onnx_checker=False
                      )


if __name__ == "__main__":
    main()
