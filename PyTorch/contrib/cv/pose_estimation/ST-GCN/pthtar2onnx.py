# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import torch
import torch.onnx
from collections import OrderedDict
from net.onnx_net.st_gcn import Model
from collections import OrderedDict


def convert():
    checkpoint = torch.load("model_best.pt", map_location='cpu')
    model = Model(in_channels=3,
                  num_class=400,
                  edge_importance_weighting=True,
                  graph_args={'layout': "openpose",
                              'strategy': "spatial"}
                  )
    model.load_state_dict(checkpoint)
    model.eval()
    print(model)
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(64, 3, 150, 18, 2)
    torch.onnx.export(model, dummy_input, "stgcn_npu.onnx", input_names=input_names, output_names=output_names,
                      opset_version=11)


if __name__ == "__main__":
    convert()
