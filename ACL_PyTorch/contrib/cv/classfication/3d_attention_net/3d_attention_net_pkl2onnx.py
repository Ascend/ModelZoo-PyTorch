# Copyright 2021 Huawei Technologies Co., Ltd
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
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel


def pkl2onnx(input_file = "model_92_sgd.pkl", output_file = "3d_attention_net.onnx"):
    model = ResidualAttentionModel()
    model.load_state_dict((torch.load(input_file, map_location = "cpu")))
    model.eval()
    input_name = ["image"]
    output_name = ["class"]
    dynamic_axes = {"image": {0:"-1"}, "class": {0:"-1"}}
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy_input, output_file, input_names = input_name, dynamic_axes = dynamic_axes, output_names = output_name, opset_version=11, verbose=True)
    
if __name__ == "__main__":
    print("----------start----------")
    pkl2onnx()
    print("----------end----------")