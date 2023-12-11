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
# limitations under the License.ls

import os
import sys
import argparse
import torch
import torch_aie

sys.path.append(r"./detr")
from hubconf import detr_resnet50_onnx

def get_args_parser():
    args = argparse.ArgumentParser(add_help=False)
    args.add_argument("--device_id", default=0, type=int)
    args.add_argument("--pre_trained", default="./model/detr.pth", type=str)
    args.add_argument("--compiled_output_dir", default="./model", type=str)
    return args

def compile_models(args, batch_size = 1):
    torch_aie.set_device(args.device_id)
    input_shape = [[768, 1280, 24, 40], [768, 768, 24, 24], [768, 1024, 24, 32], [1024, 768, 32, 24],
                   [1280, 768, 40, 24], [768, 1344, 24, 42], [1344, 768, 42, 24], [1344, 512, 42, 16],
                   [512, 1344, 16, 42]]
    for shape in input_shape:
        img_shape = [batch_size, 3, shape[0], shape[1]]
        mask_shape = [batch_size, shape[2], shape[3]]
        model_name = "detr_"+str(shape[0]) + "_" + str(shape[1]) + ".ts"
        print("start trace model, ,", model_name)
        model = detr_resnet50_onnx(pretrained=False)
        model.load_state_dict(torch.load(args.pre_trained, map_location="cpu")["model"])
        model.eval()
        traced_model = torch.jit.trace(model, (torch.rand(img_shape), torch.zeros(mask_shape, dtype=torch.bool)),
                                       strict = False)
        model_path = os.path.join(args.compiled_output_dir, model_name)
        torch_aie_model = torch_aie.compile(traced_model,
            inputs = [torch_aie.Input(img_shape), torch_aie.Input(mask_shape, dtype = torch.bool)],
            precision_policy =  torch_aie.PrecisionPolicy.FP16)
        torch.jit.save(torch_aie_model, model_path)
        print("save model success, ", model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    compile_models(args)