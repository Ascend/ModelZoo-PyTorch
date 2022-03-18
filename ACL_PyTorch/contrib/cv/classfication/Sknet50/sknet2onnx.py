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

import sys
sys.path.append(r'./PytorchInsight/classification')
from models.imagenet import sk_resnet50
import argparse
import torch
import onnx

parser = argparse.ArgumentParser(description='PyTorch Export ONNX')
parser.add_argument('--pth', default='', type=str, metavar='PATH',
                    help='path of pth file (default: none)')
parser.add_argument('--onnx', default='', type=str, metavar='PATH',
                    help='path of output (default: none)')
args = parser.parse_args()

def main():

    model = sk_resnet50()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.pth, 'cpu')
    t = model.state_dict()
    c = checkpoint['state_dict']
    for k in t:
        if k not in c:
            c[k] = t[k]
    model.load_state_dict(c)
    model.eval()

    input_names = ["image"]
    output_names = ["class"]
    dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model.module, dummy_input, args.onnx, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11, verbose=True)
    
    # delete transpose operator to improve performance
    model = onnx.load(args.onnx)
    graph = model.graph
    node = graph.node
    softmax_node_index = []
    del_group = []
    for i in range(len(node)):
        if node[i].op_type == 'Softmax':
            del_group.append((node[i-1], node[i], node[i+1], i))
    for g in del_group:
        new_input = g[0].input
        new_output = g[2].output
        new_name = g[1].name
        new_index = g[3]
        new_node = onnx.helper.make_node("Softmax", new_input, new_output, new_name, axis=1)
        for n in g[:-1]:
            graph.node.remove(n)
        graph.node.insert(new_index, new_node)
    onnx.save(model, args.onnx)
        
if __name__ == '__main__':
    main()
