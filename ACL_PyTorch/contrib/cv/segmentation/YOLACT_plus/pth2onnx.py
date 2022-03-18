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

'''
YOLACT pth权重文件转为onnx权重文件
'''
import sys
import os
sys.path.append('../')
import torch
import torch.onnx
import argparse
from data import *
from yolact import Yolact
#
set_cfg('yolact_plus_resnet50_config')
from torch.autograd import Variable


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser_pth2onnx = argparse.ArgumentParser(description='Turn YOLACT .pth module to .onnx module')

parser_pth2onnx.add_argument('--trained_model', type=str,
                             default='yolact_plus_resnet50_54_800000.pth', help='choose .pth module')

parser_pth2onnx.add_argument('--outputName', type=str,
                             default='yolact_plus', help='the name of the output onnx module')

parser_pth2onnx.add_argument('--dynamic', default=False, type=str2bool,
                             help='choose whether the output onnx module is dynamic or not')

args_pth2onnx = parser_pth2onnx.parse_args()

def removeAdd240Node(model):
    addNodeNum = 1227 #1227, 344
    addNode = model.graph.node[addNodeNum]
    model.graph.node.remove(addNode)
    for node in model.graph.node:
        if '1763' in node.input:
            assert node.input[0] == '1763' #'1763','1005'
            node.input[0] = '1761' #'1761','1003'


def optimSoftmax(model):
    from onnx import helper

    findOldSoftmaxNode = False
    for node in model.graph.node:
        if 'Softmax' in node.name:
            oldSoftmaxName = node.name
            oldSoftmaxInput = node.input[0]
            findOldSoftmaxNode = True
            break

    assert node.output[0] == 'output1'
    assert findOldSoftmaxNode

    model.graph.node.remove(node)

    TransposeNode_Pre = helper.make_node('Transpose', [oldSoftmaxInput], ['66666'],
                                         perm=[0, 2, 1], name='Transpose_Pre')

    newSoftmax = helper.make_node("Softmax", axis=1, inputs=["66666"],
                                  outputs=["88888"], name=oldSoftmaxName)

    TransposeNode_After = helper.make_node('Transpose', ['88888'], ['output1'],
                                           perm=[0, 2, 1], name="Transpose_After")

    model.graph.node.append(TransposeNode_Pre)
    model.graph.node.append(TransposeNode_After)
    model.graph.node.append(newSoftmax)

    a = model.graph.output[1].type.tensor_type.shape.dim[1]
    a.dim_param = '57744' # 57744, 19248
    b = model.graph.output[1].type.tensor_type.shape.dim[2]
    b.dim_param = '81'


def ReplaceScales(ori_list, scales_name):
    n_list = []
    for i, x in enumerate(ori_list):
        if i < 2:
            n_list.append(x)
        if i == 3:
            n_list.append(scales_name)
    return n_list

def optimresize(model):
    # 替换Resize节点
    i = 1311  #429
    n = model.graph.node[i]
    if n.op_type == "Resize":
        print("Resize", i, n.input, n.output)
        model.graph.initializer.append(
            onnx.helper.make_tensor('scales{}'.format(i), onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
        )
        newnode = onnx.helper.make_node(
            'Resize',
            name=n.name,
            inputs=ReplaceScales(n.input, 'scales{}'.format(i)),
            outputs=n.output,
            coordinate_transformation_mode='pytorch_half_pixel',
            cubic_coeff_a=-0.75,
            mode='linear',
            nearest_mode='floor'
        )
        model.graph.node.remove(n)
        model.graph.node.insert(i, newnode)
        print("replace {} index {}".format(n.name, i))
    # for i in range(401, 428):
    #     print('remove:', model.graph.node[401].name)
    #     model.graph.node.remove(model.graph.node[401])


def convert(path, pthPath):
    '''
    转换pth模型为onnx模型
    :param path: onnx模型存储路径
    :param pthPath: pth模型路径
    :return:
    '''
    yolact_net = Yolact()
    yolact_net.load_weights(pthPath, useCuda=False)
    yolact_net.exportOnnx = True
    yolact_net.eval()

    input_names = ["input.1"]

    dummy_input = Variable(
        torch.randn(1, 3, 550, 550))

    output_names = ["output0", "output1", "output2", "output3", "output4"]

    if args_pth2onnx.dynamic:
        dynamic_axes = {
            'input.1': {0: '-1'},
            'output0': {0: '-1'},
            'output1': {0: '-1'},
            'output2': {0: '-1'},
            'output3': {0: '-1'},
            'output4': {0: '-1'}
        }
        torch.onnx.export(yolact_net, dummy_input, args_pth2onnx.outputName + ".onnx",
                          verbose=True, dynamic_axes=dynamic_axes,
                          input_names=input_names,
                          output_names=output_names, opset_version=11, enable_onnx_checker=False)

    else:
        torch.onnx.export(yolact_net, dummy_input,
                          args_pth2onnx.outputName + '.onnx',
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=11, verbose=True, enable_onnx_checker=False)


if __name__ == '__main__':
    path = os.getcwd()
    pthPath = os.getcwd() + '/' + args_pth2onnx.trained_model
    convert(path, pthPath)
    import onnx

    model = onnx.load('./' + args_pth2onnx.outputName + '.onnx')
    removeAdd240Node(model)
    optimSoftmax(model)
    optimresize(model)
    onnx.save_model(model, args_pth2onnx.outputName + '.onnx')