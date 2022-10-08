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
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--input_size", default=64, type=int)
    parser.add_argument("--onnx_output", default='realesrgan-x4.onnx', type=str)
    args = parser.parse_args()
    # An instance of your model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(torch.load('experiments/pretrained_models/RealESRGAN_x4plus.pth')['params_ema'])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()

    # An example input you would normally provide to your model's forward() method
    x = torch.rand(args.bs, 3, args.input_size, args.input_size)

    # Export the model
    with torch.no_grad():
        torch_out = torch.onnx._export(model, x, args.onnx_output, opset_version=11, export_params=True)

if __name__ == "__main__":
    main()
