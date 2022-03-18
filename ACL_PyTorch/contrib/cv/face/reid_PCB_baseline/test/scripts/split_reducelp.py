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

import os
import sys
from utils.OXInterface.OXInterface import OXGraph


def main(input_model, out_path):
    oxgraph = OXGraph(input_model)
    # ReduceL2->ReduceSum
    onnx_node = oxgraph.get_oxnode_by_name('ReduceL2_122')
    onnx_node.set_op_type('ReduceSum')
    onnx_node.set_name('ReduceSum1')

    # 插入mul+sqrt节点
    oxgraph.insert_node(
            bef_node_info_list=['AveragePool_121:0', 'AveragePool_121:0'],
            aft_node_info_list=['ReduceSum1'],
            op_type='Mul',
            op_name='Mul1'
            )
    oxgraph.insert_node(
            bef_node_info_list=['ReduceSum1'],
            aft_node_info_list=['Expand_125'],
            op_type='Sqrt',
            op_name='Sqrt1'
            )

    oxgraph.save_new_model(out_path)


if __name__ == '__main__':
    input_model = sys.argv[1]
    out_path = sys.argv[2]
    out_path = os.path.abspath(out_path)
    print(input_model)
    print(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    main(input_model, out_path)
