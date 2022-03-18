# Copyright 2021 Huawei Technologies Co., Ltd
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
import sys
import torch
sys.path.append('./AlphaPose')
from alphapose.models import builder
from alphapose.utils.config import update_config


def pth2onnx(config, checkpoint, output_file):
    model = builder.build_sppe(config.MODEL, preset_cfg=config.DATA_PRESET)
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

    input_names = ["image"]
    output_names = ["output"]
    dynamic_axes = {"image": {0: "-1"},
                    "output": {0: "-1"}}
    dummy_input = torch.randn(1, 3, 256, 192)

    torch.onnx.export(model, dummy_input, output_file,
                      verbose=True, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11)


if __name__ == '__main__':
    config_file = sys.argv[1]
    checkpoint = sys.argv[2]
    output_file = sys.argv[3]
    cfg = update_config(config_file)
    pth2onnx(cfg, checkpoint, output_file)
