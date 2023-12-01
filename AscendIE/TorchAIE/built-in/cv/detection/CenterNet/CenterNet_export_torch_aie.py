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
import torchvision

def export_torch_aie(model_path, batch_size, soc_version, save_path, device_id):
    trace_model = torch.jit.load(model_path)
    trace_model.eval()
    input_info = [torch_aie.Input((batch_size, 3, 512, 512))]
    torch_aie.set_device(device_id)
    torchaie_model = torch_aie.compile(
        trace_model,
        inputs=input_info,
        allow_tensor_replace_int = True,
        torch_executed_ops = [],
        precision_policy=torch_aie.PrecisionPolicy.FP32,
        soc_version=soc_version,
        )
    suffix = os.path.splitext(model_path)[-1]
    saved_name = os.path.basename(model_path).split('.')[0] + f"b{batch_size}_torch_aie" + suffix
    torchaie_model.save(os.path.join(save_path, saved_name))
    print("[INFO] torch_aie compile for CenterNet finished, model saved in: ", os.path.join(save_path, saved_name))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-script-path', type=str, default='./CenterNet_torchscript.pt', help='trace model path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--save-path', type=str, default='./', help='compiled model path')
    parser.add_argument('--soc-version', type=str, default='Ascend310P3', help='soc version')
    parser.add_argument('--device-id', type=int, default=0, help='device id')
    opt_args = parser.parse_args()
    return opt_args

def main():
    print("[INFO] torch_aie compile for CenterNet start")
    opt_args = parse_opt()
    export_torch_aie(opt_args.torch_script_path, opt_args.batch_size, opt_args.soc_version, opt_args.save_path, opt_args.device_id)

if __name__ == '__main__':
    main()