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

import torch
import argparse
from torch.autograd import Variable


def main(args):
    model = torch.load(args.pth)
    x = torch.randn(args.batch_size, 3, 384, 128)

    with torch.no_grad():
        model.eval()

    input_names = ["input_1"]
    output_names = ["output_1"]

    # Export the model
    torch.onnx.export(model, x, args.onnx, input_names=input_names, output_names=output_names,
                      opset_version=11, verbose=False, do_constant_folding=True, export_params=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pcb")
    # data
    parser.add_argument('-p', '--pth', type=str, default='./PCB_3_7.pt', )
    parser.add_argument('-o', '--onnx', type=str, default='./PCB.onnx', )
    parser.add_argument('-b', '--batch_size', type=int, default=1, )

    main(parser.parse_args())
