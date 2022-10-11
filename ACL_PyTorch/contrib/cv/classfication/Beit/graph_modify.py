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
import sys
import numpy as np
from magiconnx import OnnxGraph


def create_slice_node(graph, idx):
    name = "Slice_" + str(idx)
    op_type = "Slice"
    dst = graph.add_node(name, op_type)
    
    return dst


def create_concat_node(graph, idx, axis):
    name = 'Concat_' + str(idx)
    op_type = "Concat"
    attrs = {
        'axis': axis,
    }
    dst = graph.add_node(name, op_type, attrs)
    return dst


def add_add_tensor(graph):
    """
    add算子中的两个输入分别为网络中的输入以及一个常量，
    由于对网络中tensor的shape进行了修改，因此同时需要对常量的shape进行修改，使其shape一致。
    """
    add_nodes = graph.get_nodes('Add')  
    if add_nodes is None or len(add_nodes) is 0:
        print('There is no Add.')
    for node in add_nodes:
        input = graph[node.inputs[1]]
        if input is not None and (input.op_type == "Constant" or input.op_type == "Initializer"):
            value = input.value.copy()
            tensor_shape = list(value.shape)
            flag = False
            for i in range(0, len(tensor_shape)):
                if tensor_shape[i] == 197:
                    tensor_shape[i] = 208
                    flag = True
            if flag: 
                tensor = np.zeros(tensor_shape, dtype=np.float32)
                for i in range(0, value.shape[0]):
                    for j in range(0, value.shape[1]):
                        for k in range(0, value.shape[2]):
                            for l in range(0, value.shape[3]):
                                tensor[i][j][k][l] = value[i][j][k][l]
                graph[node.inputs[1]].value = tensor


def refill_zero_gather(graph, batch_size):
    """
    由于将tenosr的shape从197改为208，其中新增加的11维在经过多轮计算后值发生了变化会影响后续乘法等计算，
    因此将其新增加11维中的数据重新改为0 从而避免影响
    """
    # gather shape = (bs, 12, 208, 64)   ==> (bs, 12, 197, 64)
    gather_nodes = graph.get_nodes('Gather')
    if gather_nodes is None or len(gather_nodes) is 0:
        print('There is no Gather.')
    # nums = ["49", "128", "207", "286", "365", "444", "523", "602", "681", "760", "839", "918"]      # torch 1.12.0
    idx = 6000

    for node in gather_nodes:
        if graph[node.inputs[1]].value == 1:
            concat = create_concat_node(graph, idx+1, 2)
            graph.insert_node(node.name, concat, index=0, mode='after')

            tensor = np.float32(np.zeros((int(batch_size), 12, 11, 64)))
            graph.add_initializer(str(idx+2), tensor)
            concat.inputs.append(str(idx+2))
            # ============================================================
            sli = create_slice_node(graph, idx+3)
            graph.insert_node(node.name, sli, index=0, mode='after')

            starts = np.array([0], dtype=np.int64)
            ends = np.array([197], dtype=np.int64)
            axes = np.array([2], dtype=np.int64)
            steps = np.array([1], dtype=np.int64)
            graph.add_initializer(str(idx+4), starts)
            graph.add_initializer(str(idx+5), ends)
            graph.add_initializer(str(idx+6), axes)
            graph.add_initializer(str(idx+7), steps)
            sli.inputs.append(str(idx+4))
            sli.inputs.append(str(idx+5))
            sli.inputs.append(str(idx+6))
            sli.inputs.append(str(idx+7))
            idx = idx + 8


def reshape_axis(graph: OnnxGraph) -> None:
    """
    modify 'Reshape' operator shape
    将concat后tensor的shape进行修改后，需要将网络中出现的reshape同样进行修改，
    否则reshape算子会将tensor的shape重新改回原来的shape
    """
    reshape_nodes = graph.get_nodes('Reshape')      # reshape_44  ==> concat_43
    if reshape_nodes is None or len(reshape_nodes) is 0:
        print('There is no Reshape.')
    for node in reshape_nodes:
        input = graph[node.inputs[1]]
        if input is not None and (input.op_type == "Constant" or input.op_type == "Initializer"): # and input0.op_type != 'Gather':
            newshape = input.value.copy()
            for i in range(0, len(input.value)):
                if input.value[i] == 197:
                    newshape[i] = 208
            input.value = newshape


def add_concat_tensor(graph: OnnxGraph, batch_size: int) -> None:
    """
    modify 'Concat' operator, add one tensor to input
    对add算子进行修改，使其输出tensor的shape后面两维均为16的倍数，从而避免了后续数据格式改变。
    """
    nodes = graph.get_nodes('Concat')
    node_name = nodes[0].name
    # initializer_name = '166'
    # node_name = 'Concat_18'
    # node_name = 'Concat_24'
    
    initializer_name = '1234'        # 新定义变量
    tensor = np.zeros((int(batch_size), 11, 768), dtype=np.float32)
    
    graph.add_initializer(initializer_name, tensor)
    graph[node_name].inputs.append(initializer_name)


def del_layernorm_transdata(graph: OnnxGraph, batch_size: int) -> bool:
    reshape_axis(graph)
    add_concat_tensor(graph, int(batch_size))
    add_add_tensor(graph)
    refill_zero_gather(graph, int(batch_size))
    return True


def improve_model(path: str, new_path: str, batch_size: int) -> None:
    graph = OnnxGraph(path)
    if graph is None:
        print('onnx model not exist.')
        return None
    ret = del_layernorm_transdata(graph, batch_size)
    if not ret:
        print('delete layernorm transdata failed.')
        return None
    graph.save(new_path)


def dump_data(path, save_path, batch_size: int):
    graph = OnnxGraph(path)
    data = np.random.randn(batch_size,3,224,224)
    data = data.astype(np.float32)
    graph.dump([data,], save_path)


if __name__ == '__main__':

    if len(sys.argv) is not 4:
        print('only need 3 params, include: onnx source path | dest path | batch_size.')
    else:
        improve_model(sys.argv[1], sys.argv[2], sys.argv[3])

