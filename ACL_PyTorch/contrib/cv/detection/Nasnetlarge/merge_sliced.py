"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('./test/onnx_tools')
from OXInterface.OXInterface import OXGraph


def mergeIntializer(oxgraph, initializer1, initializer2):
    """
    :param oxgraph: input onnx graph
    :param initializer1: initializer need to be merged
    :param initializer2: initializer need to be merged
    :return: merged initializer
    """
    merged_name = '{}_{}'.format(initializer1.get_name(), initializer2.get_name())
    merged_data = np.append(
        initializer1.get_data(),
        initializer2.get_data(),
    )
    try:
        return oxgraph.get_oxinitializer_by_name(merged_name)
    except RuntimeError:
        print("Insert a new initializer.")
        merged_initializer = oxgraph.add_initializer(merged_name, merged_data)
        return merged_initializer


def mergeSlicedOp(oxgraph, slice_node1, slice_node2):
    """
    :param oxgraph: input onnx graph
    :param slice_node1: slice node1 need to be merged
    :param slice_node2: slice node2 need to be merged
    :return: merged graph
    """
    # modify slice_node1 -> merge_node
    slice_node1.input[1] = mergeIntializer(
        oxgraph,
        oxgraph.get_oxinitializer_by_name(slice_node1.input[1]),
        oxgraph.get_oxinitializer_by_name(slice_node2.input[1])).get_name()
    slice_node1.input[2] = mergeIntializer(
        oxgraph,
        oxgraph.get_oxinitializer_by_name(slice_node1.input[2]),
        oxgraph.get_oxinitializer_by_name(slice_node2.input[2])).get_name()
    slice_node1.input[3] = mergeIntializer(
        oxgraph,
        oxgraph.get_oxinitializer_by_name(slice_node1.input[3]),
        oxgraph.get_oxinitializer_by_name(slice_node2.input[3])).get_name()
    slice_node1.input[4] = mergeIntializer(
        oxgraph,
        oxgraph.get_oxinitializer_by_name(slice_node1.input[4]),
        oxgraph.get_oxinitializer_by_name(slice_node2.input[4])).get_name()
    oxgraph.remove_node(slice_node2.get_name())
    return oxgraph


def getContinuousOp(oxgraph, op_type='Slice'):
    """
    :param oxgraph: input onnx graph
    :param op_type: op_type to be searched
    :return: continuous op list
    """
    all_slice_ops = oxgraph.get_oxnode_by_op_type(op_type)
    flags = [-1] * len(all_slice_ops)
    res = []
    for idx, node in enumerate(all_slice_ops):
        next_node = oxgraph.get_next_oxnode(node.get_name())[0]
        if next_node in all_slice_ops:
            next_idx = all_slice_ops.index(next_node)
            if flags[idx] == -1 and flags[next_idx] == -1:
                res.append([node, next_node])
                flags[idx] =  flags[next_idx] = len(res) - 1
            elif flags[idx] != -1 and flags[next_idx] == -1:
                res[flags[idx]].append(next_node)
                flags[next_idx] = flags[idx]
            elif flags[idx] == -1 and flags[next_idx] != -1:
                res_idx = res[flags[next_idx]].index(next_node)
                res[flags[next_idx]].insert(res_idx, node)
                flags[idx] = flags[next_idx]
            else:
                res[flags[idx]] = res[flags[idx]] + res[flags[next_idx]]
                flags[next_idx] = flags[idx]
    flags = list(filter(lambda x: x != -1, flags))
    uniq_flags = []
    for f in flags:
        if f not in uniq_flags:
            uniq_flags.append(f)
    return [res[idx] for idx in uniq_flags]


def main(model_path, out_model):
    """main function"""
    oxgraph = OXGraph(model_path)
    continuous_slice_nodes = getContinuousOp(oxgraph)
    for nodes in tqdm(continuous_slice_nodes):
        if len(nodes) > 2:
            raise NotImplementedError()
        slice_node1, slice_node2 = nodes
        oxgraph = mergeSlicedOp(oxgraph, slice_node1, slice_node2)
    oxgraph.save_new_model(out_model)


if __name__ == '__main__':
    input_path = sys.argv[1]
    out_path = sys.argv[2]
    main(input_path, out_path)
