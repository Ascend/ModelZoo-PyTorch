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

import sys
import os
import numpy as np
from auto_optimizer import OnnxGraph


def remove_node_and_print(_graph, node):
    print(f"Removed: {node.name}")
    _graph.remove(node.name)
    

def get_resize_node_after_conv(_graph, conv_node):
    ''' Conv -> Relu -> [Resize] '''
    relu_node = _graph.get_next_nodes(conv_node.outputs[0])[0]
    if relu_node.op_type != "Relu":
        return None
    
    relu_next_nodes = _graph.get_next_nodes(relu_node.outputs[0])
    for next_node in relu_next_nodes:
        if next_node.op_type == "Resize":
            return next_node

    return None


def remove_or_modify_resize(_graph):
    for conv_node in _graph.get_nodes("Conv"):
        conv_weight = conv_node.inputs[1]
        conv_weight_shape = _graph[conv_weight].value.shape
        
        if conv_weight_shape == (256, 32, 1, 1):
            '''Remove'''
            resize_node = get_resize_node_after_conv(_graph, conv_node)
            remove_node_and_print(_graph, resize_node)
        
        elif conv_weight_shape == (256, 64, 1, 1) or conv_weight_shape == (256, 160, 1, 1):
            '''Change mode to nearest'''
            resize_node = get_resize_node_after_conv(_graph, conv_node)
            if resize_node:
                print(f"Modified: {resize_node.name}")
                resize_node["mode"] = b"nearest"

    '''Remove the last Resize node'''
    resize_node = _graph.get_nodes("Resize")[-1]
    remove_node_and_print(_graph, resize_node)
    return _graph


def remove_unsqueeze(_graph):
    '''Remove the last Unsqueeze node'''
    unsqueeze_node = _graph.get_nodes("Unsqueeze")[-1]
    remove_node_and_print(_graph, unsqueeze_node)
    return _graph


def insert_cast(_graph):
    '''Insert Cast to make the model output in the INT32 format'''
    argmax_node = _graph.get_nodes("ArgMax")[0]
    new_cast_node = _graph.add_node('Cast_new', 'Cast', attrs={'to': 6})  # 6 -> INT32
    print(f"Inserted: {new_cast_node.name} after {argmax_node.name}")
    _graph.insert_node(argmax_node.name, new_cast_node)
    return _graph


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("usage: python3 xxx.py [src_path] [save_path]")
    in_onnx = sys.argv[1]
    out_onnx = sys.argv[2]
    in_onnx = os.path.realpath(in_onnx)
    out_onnx = os.path.realpath(out_onnx)
    
    onnx_graph = OnnxGraph.parse(in_onnx)
    onnx_graph = remove_or_modify_resize(onnx_graph)
    onnx_graph = remove_unsqueeze(onnx_graph)
    onnx_graph = insert_cast(onnx_graph)

    onnx_graph.outputs[0].dtype = np.dtype("int32")
    print("Modify output dtype to 'int32'")

    onnx_graph.save(out_onnx)