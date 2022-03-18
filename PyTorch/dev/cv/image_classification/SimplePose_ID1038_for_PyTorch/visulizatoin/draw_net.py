#
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
#
from graphviz import Digraph
from torch.autograd import Variable
import torch


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (T O D O: make optional)
    """
    # visulize the netwok or drwa the network or show the network
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(
                    var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


if __name__ == '__main__':

    # output the shape of of every layers' feature maps
    from models.layers_transposed import Conv, Residual, Hourglass, SELayer
    from models.posenet import PoseNet, NetworkEval
    from config.config import GetConfig, COCOSourceConfig, TrainingOpt

    # ##############################################################3
    # from torchsummary import summary
    #
    # device = torch.device("npu" if torch.npu.is_available() else "cpu")  # PyTorch v0.4.0
    #
    # model = Hourglass2(2, 32, 1, Residual).to(device)
    # t = model._make_hour_glass()
    # for i in t.named_modules():
    #     print(i)
    # summary(model, (32, 128, 128))
    # ##############################################################3

    # ##############################################################3

    #
    # # plot the models
    # model = Hourglass2(4, 32, 1, Conv)
    # x = Variable(torch.randn(1, 32, 128, 128))  # x鐨剆hape涓(batch锛宑hannels锛宧eight锛寃idth)
    # y = model(x)
    # g = make_dot(y)
    # g.view()
    # ##############################################################3

    import torch.onnx

    net = Hourglass(4, 256, 128, resBlock=Conv, bn=True)
    dummy_input = torch.randn(1, 256, 128, 128)
    torch.onnx.export(net, dummy_input, "hourglass.onnx")
    #
    # se = SELayer(256)
    # dummy_input = torch.randn(8, 256, 128, 128)
    # torch.onnx.export(se, dummy_input, "SElayer.onnx")
    #
    # opt = TrainingOpt()
    # config = GetConfig(opt.config_name)
    # pose = NetworkEval(opt, config, bn=True)
    # pose.eval()
    #
    # # # ######################### Visualize the network ##############
    # dummy_input = torch.randn(1, 512, 512, 3)
    # y = pose(dummy_input)[0][0]
    # print(y.shape)
    # # # ############################# netron --host=localhost --port=8080
    # # torch.onnx.export(pose, dummy_input, "posenet.onnx")
    # # # export onnx for the second time to check the model
    # # torch.onnx.export(pose, dummy_input, "posenet2.onnx")
    #
    # # # ##############  Count the FLOPs of your PyTorch model  ##########################
    # from thop import profile
    # from thop import clever_format
    # flops, params = profile(pose, inputs=(dummy_input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)