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

import torch

sys.path.append(r'./SRFlow/code')
from models import create_model
import options.options as option


def load_model(cfg_path, ckpt_path):
    opt = option.parse(cfg_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)
    md = torch.load(ckpt_path)
    for name in md:
        if name.endswith('invconv.weight'):
            md[name] = torch.inverse(md[name].double()).float()
    inv_model_path = ckpt_path[:-4] + '_inv.pth'
    torch.save(md, inv_model_path)
    model.load_network(load_path=inv_model_path, network=model.netG)
    return model, opt


def main():
    srflowmodel, _ = load_model(args.cfg, args.pth)
    model = srflowmodel.netG
    x = torch.randn(1, 3, 256, 256)
    model.eval()
    input_names = ["input-image"]
    output_names = ["output-image"]
    dynamic_axes = {'input-image': {0: '-1'}, 'output-image': {0: '-1'}}
    # Export the model
    torch.onnx.export(model, x, args.onnx, 
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=11, 
                      verbose=False, 
                      do_constant_folding=True,
                      export_params=True,
                      enable_onnx_checker=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='PyTorch convert to ONNX')
    parser.add_argument('--cfg', type=str, metavar='PATH',
                        default='./SRFlow/code/confs/SRFlow_DF2K_8X.yml',
                        help='path of pytorch checkpoint file (default: none)')
    parser.add_argument('--pth', type=str, metavar='PATH', required=True,
                        help='path of pytorch checkpoint file (default: none)')
    parser.add_argument('--onnx', default='', type=str, metavar='PATH',
                        help='path of output onnx model (default: none)')
    args = parser.parse_args()
    main()
