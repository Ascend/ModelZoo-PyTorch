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

import sys
from auto_optimizer import OnnxGraph

inp = sys.argv[1]
out = sys.argv[2]

g = OnnxGraph.parse(inp)

to_del_nodes = [
	'NonMaxSuppression_683', 'Slice_688', 'Gather_690', 'Slice_695', 'Gather_697',
	'Reshape_699', 'Shape_700', 'Gather_702', 'Mul_703', 'Add_704', 'Cast_705', 
	'Gather_706', 'Shape_707', 'Gather_709', 'Unsqueeze_710', 'Concat_712', 
	'Cast_713', 'ReduceMin_714', 'Cast_715', 'Unsqueeze_716', 'TopK_717', 
	'Squeeze_719', 'Gather_720', 'Slice_725', 'Cast_726', 'Gather_727', 'Gather_729', 
	'Unsqueeze_730', 'Gather_733', 'Unsqueeze_bboxes', 'Unsqueeze_scores'
]
for node_name in to_del_nodes:
	g.remove(node_name, mapping={})

# =========================== add new nodes =============================
new_node = g.add_node(
	'Unsqueeze_new_0', 
	'Unsqueeze',
	outputs=['Unsqueeze_new_0_out_0'],
	attrs={'axes': [2]}
)
g.insert_node('Concat_659', new_node, refer_index=0, mode='after')

new_node = g.add_node(
	'Transpose_new_0', 
	'Transpose',
	outputs=['Transpose_new_0_out_0'],
	attrs={'perm': [0, 2, 1]}
)
g.insert_node('Slice_676', new_node, refer_index=0, mode='after')
new_node = g.add_node(
	'Cast_new_0', 
	'Cast',
	outputs=['Add_new_0_out_0'],
	attrs={'to': 7}
)
g.insert_node('Add_labels', new_node, refer_index=0, mode='before')

new_node = g.add_node(
	'BatchMultiClassNMS_new_0', 
	'BatchMultiClassNMS',
	attrs=dict(
		iou_threshold=0.5,
	    max_size_per_class=200,
	    max_total_size=200,
	    score_threshold=0.05
	)
)
g.connect_node(
	new_node,
	['Unsqueeze_new_0', 'Transpose_new_0'],
	['bboxes', 'scores', 'Cast_new_0']
)

g.remove_unused_nodes()
g.update_map()
g.toposort()
g.save(out)
