# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from onnx import helper, TensorProto
from auto_optimizer import OnnxGraph
from auto_optimizer.graph_refactor.interface import Initializer
from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer

model = "./yolox.onnx"
graph = OnnxGraph.parse(model)
print(f"load {model} success!")
outputs = graph.outputs

# 获取原模型输出节点前的transpose节点，由于该改图操作是在修改torch模型结构的基础上进行的改图，故无法通过整图输出节点的位置获取该transpose,需要手动检索
transpose_output = graph["Transpose_333"].outputs
next_nodes = graph.get_next_nodes(graph["Transpose_333"].outputs[0])
# cast to fp32
graph.add_node("New_Cast", "Cast", inputs=[transpose_output[0]], outputs=["transpose_cast_to_fp32"], attrs={"to": 1})
for i in next_nodes:
    graph[i.name].inputs[0] = "transpose_cast_to_fp32"

# 开源模型在该步骤transpose之前的conf.shape为[bs, 8400, 1]，需要将conf transpose为nms要求的[bs, 1, 8400]
graph.add_node("transpose_conf", "Transpose", inputs=[outputs[1].name], outputs=["scores_transpose"],
               attrs={"perm": [0, 2, 1]})  # output shape [1, 1, 8400]

# NonMaxSuppression
graph.add_initializer("iou_threshold", np.array([0.65]).astype(np.float32))  # 0.65
graph.add_initializer("score_threshold", np.array([0.001]).astype(np.float32))  # 0.001
graph.add_initializer("max_output_boxes_per_class", np.array([400]).astype(np.int64))  # 该数值需要调测，取不影响精度的最小值，数值过大会使性能劣化

# "center_point_box": 0 (左上x, 左上y, 右下x, 右下y); 1 (x_center, y_center, width, height)
a = graph.add_node("new_nms", "NonMaxSuppression",
                   inputs=[outputs[3].name, "scores_transpose", "max_output_boxes_per_class", "iou_threshold",
                           "score_threshold"], attrs={"center_point_box": 0},
                   outputs=["indices"])

b = graph.add_node("concat_bbox_scores_cls", "Concat", inputs=[outputs[0].name, outputs[1].name, outputs[2].name],
                   outputs=["prediction"], attrs={"axis": 2})

for i in [outputs[0].name, outputs[1].name, outputs[2].name, outputs[3].name]:
    graph.remove(i)

graph.add_output(a.outputs[0], "int64", None)
graph.add_output(b.outputs[0], "float32", None)

graph.infershape()
model_name = "yolox_modify.onnx"
graph.save(model_name)
print(f"{model_name} add nms_op finished")
