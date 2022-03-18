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

import sys
import torch
from collections import OrderedDict
sys.path.append(r"./3DMPPE_ROOTNET_RELEASE")
from main.model import get_pose_net
from main.config import cfg

def convert(pth_file_path,onnx_file_path):
    model = get_pose_net(cfg, False)
    ckpt = torch.load(pth_file_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in ckpt['network'].items():
        if k[0:7] == "module.":
            name = k[7:]  # remove module.
        else:
            name = k[0:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    input_names = ["image", "cam_param"]
    output_names = ["score"]
    dynamic_axes = {'image': {0: '-1'}, 'cam_param': {0: '-1'}, 'score': {0: '-1'}}
    dummy_input1 = torch.randn(1, 3, 256, 256)
    dummy_input2 = torch.randn(1, 1)
    torch.onnx.export(model, (dummy_input1, dummy_input2), onnx_file_path, input_names=input_names, dynamic_axes=dynamic_axes,
                      output_names=output_names, opset_version=11, verbose=True)

if __name__ == "__main__":
    # convert("snapshot_6.pth.tar", "3DMPPE-ROOTNET.onnx")
    convert(sys.argv[1], sys.argv[2])


