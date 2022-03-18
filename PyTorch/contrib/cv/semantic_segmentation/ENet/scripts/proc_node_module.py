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

import torch
from collections import OrderedDict
import os

def proc_nodes_module(checkpoint):
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if "module." in k:
            name = k.replace("module.", "")
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

root ='~/.torch/models'
root = os.path.expanduser(root)
file_path = os.path.join(root, 'enet_citys.pth')
checkpoint = torch.load(file_path, map_location='cpu')
checkpoint = proc_nodes_module(checkpoint)


#directory = os.path.expanduser(args.save_dir)
directory = os.path.expanduser(root)
filename = 'enet_citys.pth'
filename = os.path.join(directory, filename)
torch.save(checkpoint, filename)
