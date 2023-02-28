# Copyright 2023 Huawei Technologies Co., Ltd
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

import argparse

import onnx
import numpy as np

from auto_optimizer import OnnxGraph
from auto_optimizer.graph_refactor.interface.base_node import Initializer


def insert_cast_op(graph):
    '''
    om的ScatterND和Slice算子不支持int64，因此在onnx此算子前后插入Cast算子，实现：int64 -> int32 -> int64
    '''
    scatter_nodes = graph.get_nodes(op_type="ScatterND")
    for scatter_node in scatter_nodes:
        for idx, inp in enumerate(scatter_node.inputs):
            if graph.get_node(inp, Initializer):
                graph[inp].value = graph[inp].value.astype("int32")
            else:
                cast_bef = graph.add_node(
                    f"Cast_bef_{inp}",
                    'Cast',
                    attrs={'to':int(onnx.TensorProto.INT32)}
                )
                graph.insert_node(scatter_node.name, cast_bef, refer_index=idx, mode="before")
        cast_aft = graph.add_node(
            f"Cast_aft_{scatter_node.name}",
            'Cast',
            attrs={'to':int(onnx.TensorProto.INT64)}
        )
        graph.insert_node(scatter_node.name, cast_aft, mode="after")
    
    slice_nodes = graph.get_nodes(op_type="Slice")
    for slice_node in slice_nodes:
        for idx, inp in enumerate(slice_node.inputs):
            if graph.get_node(inp, Initializer):
                graph[inp].value = graph[inp].value.astype("int32")
            else:
                cast_bef = graph.add_node(
                    f"Cast_bef_{slice_node.name}",
                    'Cast',
                    attrs={'to':int(onnx.TensorProto.INT32)}
                )
                graph.insert_node(slice_node.name, cast_bef, refer_index=idx, mode="before")
        cast_aft = graph.add_node(
            f"Cast_aft_{slice_node.name}",
            'Cast',
            attrs={'to':int(onnx.TensorProto.INT32)}
        )
        graph.insert_node(slice_node.name, cast_aft, mode="after")


def process_gather_op_indices(graph):
    '''
    om的GatherV2D算子的indices不支持-1输入，因此需要处理onnx中indices为-1的Gather算子
    '''
    gather_nodes = graph.get_nodes(op_type="Gather")
    for gather_node in gather_nodes:
        indice_init = graph.get_node(gather_node.inputs[1], Initializer)
        if indice_init and indice_init.value == -1:
            data_value_info = graph._value_map[gather_node.inputs[0]]
            gather_init = graph.add_initializer(
                f"{gather_node.name}_init",
                np.array(data_value_info.shape[0] - 1, dtype="int64")
            )
            gather_node.inputs[1] = gather_init.name


def fix_topk_op(graph):
    '''
    删除TopK算子的后继冗余算子reshape-expand
    '''
    topk_nodes = graph.get_nodes(op_type="TopK")
    for topk_node in topk_nodes:
        next_node = graph.get_next_nodes(topk_node.outputs[1])[0]
        if next_node.op_type == "Reshape":
            next_next_node = g.get_next_nodes(next_node.outputs[0])[0]
            if next_next_node.op_type == "Expand":
                graph.remove(next_node.name)
                graph.remove(next_next_node.name)


if __name__ == "__main__":
    """
    Usage Example:
    python3 modify_onnx.py \
    --input_model_path ./model/transformer_greedySearch_input15_maxSeqLen15_sim.onnx \
    --output_model_path ./model/transformer_greedySearch_input15_maxSeqLen15_sim_mod.onnx
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', required=True)
    parser.add_argument('--output_model_path', required=True)
    opt = parser.parse_args()

    g = OnnxGraph.parse(opt.input_model_path)
    insert_cast_op(g)
    process_gather_op_indices(g)
    fix_topk_op(g)
    g.save(opt.output_model_path)


