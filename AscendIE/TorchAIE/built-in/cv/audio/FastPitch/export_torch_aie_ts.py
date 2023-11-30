# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
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
import os
import argparse
import torch
import torch_aie
from torch_aie import _enums

def export_torch_aie(opt_args):
    trace_model = torch.jit.load(opt_args.torch_script_path)
    trace_model.eval()

    torch_aie.set_device(0)
    inputs = []
    inputs.append(torch_aie.Input((opt_args.batch_size, 200)))
    torchaie_model = torch_aie.compile(
        trace_model,
        inputs=inputs,
        precision_policy=_enums.PrecisionPolicy.FP16,
        truncate_long_and_double=True,
        require_full_compilation=False,
        allow_tensor_replace_int=False,
        min_block_size=3,
        torch_executed_ops=[],
        soc_version=opt_args.soc_version,
        optimization_level=0)
    suffix = os.path.splitext(opt_args.torch_script_path)[-1]
    saved_name = os.path.basename(opt_args.torch_script_path).split('.')[0] + f"_torch_aie_bs{opt.batch_size}" + suffix
    torchaie_model.save(os.path.join(opt_args.save_path, saved_name))
    print("torch aie yolov3 compiled done. saved model is ", os.path.join(opt_args.save_path, saved_name))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_script_path', type=str, default='./fastpitch.torchscript.pt', help='trace model path')
    parser.add_argument('--soc_version', type=str, default='Ascend310P3', help='soc version')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--save_path', type=str, default='./', help='compiled model path')
    opt_args = parser.parse_args()
    return opt_args

def main(opt_args):
    export_torch_aie(opt_args)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)