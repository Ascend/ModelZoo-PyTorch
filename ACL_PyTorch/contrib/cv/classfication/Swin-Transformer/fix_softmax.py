# Copyright 2020 Huawei Technologies Co., Ltd
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
from onnx_tools.oxinterface.OXInterface import OXGraph


def fix_softmax(oxgraph, node):
    # get pre && after nodes
    pre_nodes = [_.get_name() for _ in oxgraph.get_previous_oxnode(node.get_name())]
    aft_nodes = [_.get_name() for _ in oxgraph.get_next_oxnode(node.get_name())]
    # insert transpose
    oxgraph.insert_node(
        bef_node_info_list=pre_nodes,
        aft_node_info_list=[node.get_name()],
        op_type='Transpose',
        op_name='Transpose_before_{}'.format(node.get_name()),
        perm=(0, 3, 1, 2)
    )

    # fix attr
    node.set_attribute(attr_name='axis', attr_value=1)

    # insert transpose
    oxgraph.insert_node(
        bef_node_info_list=[node.get_name()],
        aft_node_info_list=aft_nodes,
        op_type='Transpose',
        op_name='Transpose_after_{}'.format(node.get_name()),
        perm=(0, 2, 3, 1)
    )


def main(model_path, save_path):
    oxgraph = OXGraph(model_path)
    softmax_nodes = oxgraph.get_oxnode_by_op_type(op_type='Softmax')
    for node in softmax_nodes:
        fix_softmax(oxgraph, node)
    oxgraph.save_new_model(save_path)


if __name__ == '__main__':
    input_path = sys.argv[1]
    out_path = sys.argv[2]
    main(input_path, out_path)
