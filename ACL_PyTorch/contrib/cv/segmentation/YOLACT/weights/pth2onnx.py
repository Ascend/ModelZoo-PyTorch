# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
'''
YOLACT pth权重文件转为onnx权重文件
'''
import sys
import os
sys.path.append('../')
import torch
import torch.onnx
import argparse
from yolact import Yolact
from torch.autograd import Variable

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser_pth2onnx = argparse.ArgumentParser(description='Turn YOLACT .pth module to .onnx module')

parser_pth2onnx.add_argument('--cann_version', type=str, 
        default='6', help='choose cann_version')
parser_pth2onnx.add_argument('--trained_model', type=str, 
        default='yolact_base_55_410000.pth', help='choose .pth module')

parser_pth2onnx.add_argument('--outputName', type=str, 
        default='yolact', help='the name of the output onnx module')

parser_pth2onnx.add_argument('--dynamic', default=True, type=str2bool, 
        help='choose whether the output onnx module is dynamic or not')

args_pth2onnx = parser_pth2onnx.parse_args()


def removeAdd240Node(model):
    if torch.__version__ == '1.8.0':
        addNodeNum = 240
    else:
        addNodeNum = 344
    addNode = model.graph.node[addNodeNum]
    model.graph.node.remove(addNode)
    for node in model.graph.node:
        if '1005' in node.input:
            assert node.input[0] == '1005'
            node.input[0] = '1003'


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
    a.dim_param = '19248'
    b = model.graph.output[1].type.tensor_type.shape.dim[2]
    b.dim_param = '81'


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
            'input.1':{0:'-1'},
            'output0':{0:'-1'},
            'output1':{0:'-1'},
            'output2':{0:'-1'},
            'output3':{0:'-1'},
            'output4':{0:'-1'}
        }
        torch.onnx.export(yolact_net, dummy_input, args_pth2onnx.outputName + ".onnx",
                          verbose=True, dynamic_axes=dynamic_axes,
                          input_names=input_names,
                          output_names=output_names, opset_version=11)

    else:
        torch.onnx.export(yolact_net, dummy_input,
                          args_pth2onnx.outputName + '.onnx',
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=11, verbose=True)

if __name__ == '__main__':
    path = os.getcwd()
    pthPath = os.getcwd() + '/' + args_pth2onnx.trained_model
    convert(path, pthPath)
    import onnx
    if args_pth2onnx.cann_version == '5':
        model = onnx.load('./' + args_pth2onnx.outputName + '.onnx')
        removeAdd240Node(model)
        optimSoftmax(model)
        onnx.save_model(model, args_pth2onnx.outputName + '.onnx')

