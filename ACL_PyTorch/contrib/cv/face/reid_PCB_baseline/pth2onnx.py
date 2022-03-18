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
    x = torch.randn(1, 3, 384, 128)
    model.eval()
    input_names=["input_1"]
    output_names=["output_1"]
    dynamic_axes = {'input_1': {0: '-1'}, 'output_1': {0: '-1'}}
    x = Variable(x, volatile=True)
    # Export the model
    torch.onnx.export(model, x, "./models/PCB.onnx", input_names=input_names, output_names=output_names,   \
         dynamic_axes=dynamic_axes, opset_version=11, verbose=True, do_constant_folding=True, export_params=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-p', '--pth', type=str, default='./models/PCB_3_7.pt',)
    main(parser.parse_args())
