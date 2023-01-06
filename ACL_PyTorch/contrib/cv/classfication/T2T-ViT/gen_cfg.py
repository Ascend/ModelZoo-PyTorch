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
import tempfile
from auto_optimizer import OnnxGraph


def gen_cfg(graph, cfg_file):
    """Generate cfg file for '--keep_dtype' parameter"""
    node_list = []
    start_node_lists, end_node_lists = find_pattern(graph)
    
    for i, start_node in enumerate(start_node_lists):
        end_node = end_node_lists[i]
        with tempfile.TemporaryDirectory() as tmpdirname:
            pattern = graph.extract(
                os.path.join(tmpdirname, 'model.onnx'), 
                start_node.inputs, 
                end_node.outputs,
                enable_model_check=False
                )
        node_names = [node.name for node in pattern.nodes]
        node_list.extend(node_names)

    with open(cfg_file, 'w') as f:
        for node in node_list:
            f.write("{}\n".format(node))


def find_pattern(graph):
    """Find patterns with start nodes and end nodes"""
    start_nodes = []
    end_nodes = []
    
    for split_node in graph.get_nodes('Split'):
        if len(split_node.outputs) == 3:
            start_nodes.append(split_node)
    
    for reducesum_node in graph.get_nodes('ReduceSum'):
        next_node = graph.get_next_nodes(reducesum_node.outputs[0])[0]
        if next_node.op_type == 'Div':
            end_nodes.append(next_node)
    
    return start_nodes, end_nodes
  

if __name__=="__main__":
    onnx_file_name = sys.argv[1]
    cfg_file_name = sys.argv[2]
    g = OnnxGraph.parse(onnx_file_name)
    gen_cfg(g, cfg_file_name)
