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

import argparse
import os

import torch 
from cldm.model import create_model, load_state_dict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument( 
        "--model",
        type=str,
        default="./models/control_sd15_canny.pth",
        help="Path or name of the pre-trained model.",
    )
    parser.add_argument(
        "--control_path",
        type=str,
        default="./control.onnx",
        help="path or name of the control.",
    )
    parser.add_argument(
        "--sd_path",
        type=str,
        default="./sd.onnx",
        help="Path or name of the sd.",
    )

    return parser.parse_args()


def export_control(model, control_path):
    model = model.control_model.eval()
    dummy_input =(
              torch.randn(1, 4, 64, 72),
              torch.randn(1, 3, 512, 576),
              torch.tensor([1]),
              torch.randn(1, 77, 768)
              )

    torch.onnx.export(model, dummy_input, control_path, 
                      input_names = ["text", "hint", "t","cond_text"],
                      output_names = ["text_outs"], verbose=False, export_params=True,
                      opset_version=13)


def export_sd(model, sd_path):
    model = model.model.diffusion_model.eval()
    dummy_input = (
               torch.randn(1, 4, 64, 72),
               torch.tensor([1]),
               torch.randn(1, 77, 768),
               [torch.randn([1, 320, 64, 72]), torch.randn([1, 320, 64, 72]), 
               torch.randn([1, 320, 64, 72]), torch.randn([1, 320, 32, 36]), 
               torch.randn([1, 640, 32, 36]), torch.randn([1, 640, 32, 36]),
               torch.randn([1, 640, 16, 18]), torch.randn([1, 1280, 16, 18]), 
               torch.randn([1, 1280, 16, 18]), torch.randn([1, 1280, 8, 9]),
               torch.randn([1, 1280, 8, 9]), torch.randn([1, 1280, 8, 9]),
               torch.randn([1, 1280, 8, 9])]
               )

    torch.onnx.export(model, dummy_input, sd_path, 
                      input_names = ["text", "t", "cond_text",
                                     "input1", "input2", "input3", "input4",
                                     "input5", "input6", "input7", "input8",
                                     "input9", "input10", "input11", "input12", 
                                     "input13"],
                      output_names = ["text_outs"],
                      verbose=False, export_params=True, opset_version=13)


def main():
    args = parse_arguments()
    model = create_model("./models/cldm_v15.yaml" ).cpu()
    model.load_state_dict(load_state_dict(args.model, location='cpu'))
    if not os.path.exists(args.control_path):
        os.makedirs(args.control_path, mode=0o744)
    if not os.path.exists(args.sd_path):
        os.makedirs(args.sd_path, mode=0o744)
    export_control(model, os.path.join(args.control_path, "control.onnx"))
    print("control model done")
    export_sd(model, os.path.join(args.sd_path, "sd.onnx"))
    print("sd model done")
    

if __name__ == "__main__":
    main()