# Copyright 2022 Huawei Technologies Co., Ltd
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

from magiconnx import OnnxGraph
import onnx
import sys
import os

def remove_node(_graph):
    '''remove nodes'''
    invalid_nodes = ["Resize_1428", "Resize_2925", "Resize_4422", "Resize_4567", "Unsqueeze_4574"]
    for node_name in invalid_nodes:
        in_rename_map = {}
        for node_id, node in enumerate(_graph.node):
            if node.name == node_name:
                in_name = node.input[0]
                out_name = node.output[0]
                in_rename_map = {out_name: in_name}
                del _graph.node[node_id]
                break
    
        for node_id, node in enumerate(_graph.node):
            for in_id, in_name in enumerate(node.input):
                if in_name in in_rename_map:
                    node.input[in_id] = in_rename_map[in_name]
        print("[info]remove: ", node_name)
    return _graph


def insert_cast(_graph):
    '''insert cast change output to int32'''
    nodes = _graph.node
    for i in range(len(nodes)):
        if nodes[i].op_type == 'ArgMax':
            argmax = nodes[i]
            argmax_id = i
    cast_input0 = "cast_in"
    cast_output0 = "output"
    cast_new = onnx.helper.make_node(
        "Cast",
        inputs=[cast_input0],
        outputs=[cast_output0],
        name="Cast_new",
        to=getattr(onnx.TensorProto, "INT32"))
    argmax.output[0] = cast_input0
    _graph.node.insert(argmax_id + 1, cast_new)
    print("[info]insert: Cast_new op after ", argmax.name)
    return _graph


def resize_optimize(resize_nodes, speeds, accs):
    '''Resize mode to nearest'''
    for node in resize_nodes:
        # print("node.name={}".format(node.name))   
        if node['mode'] == b'linear' and node.name not in speeds and node.name not in accs:
            print("[info]{} mode to 'nearest'".format(node.name))  
            node['mode'] = b'nearest'


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("usage: python3 xxx.py [src_path] [save_path]")
    in_onnx = sys.argv[1]
    out_onnx = sys.argv[2]
    in_onnx = os.path.realpath(in_onnx)
    out_onnx = os.path.realpath(out_onnx)
    onnx_model = onnx.load(in_onnx)
    graph = onnx_model.graph
    graph = remove_node(graph)
    graph = insert_cast(graph)
    onnx.save(onnx_model, out_onnx)
    onnx_graph = OnnxGraph(out_onnx)

    phs = onnx_graph.get_nodes("Placeholder")
    phs[1].dtype = 'int32'
    print("[info]modify output detype to 'int32'")

    _speeds = ["Resize_1514", "Resize_3011", "Resize_4508"]
    _accs = ["Resize_1491", "Resize_2988", "Resize_4485"]
    _resize_nodes = onnx_graph.get_nodes(op_type='Resize')
    resize_optimize(_resize_nodes, _speeds, _accs)
    onnx_graph.save(out_onnx)