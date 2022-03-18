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
import os
import argparse
from acl_infer.acl_net import AclNet, init_acl, release_acl
import acl
import numpy as np

if __name__ == '__main__':
    DEVICE = 0
    init_acl(DEVICE)
    shapes = [262, 326, 390, 454, 518, 582, 646, 710, 774, 838, 902, 966, 1028, 1284, 1478]
    shape_num = [96, 682, 1260, 1230, 1052, 940, 656, 462, 303, 207, 132, 67, 38, 48, 3]
    shape_t = []
    RES = 0
    FPS = 0
    net = AclNet(model_path="encoder_262_1478.om", device_id=DEVICE)
    for shape in shapes:
        TEMP = 0
        memory = np.random.random((shape, 83)).astype("float32")
        dims = {'dimCount': 2, 'name': '', 'dims': [shape, 83]}
        for i in range(20):
            output_data, exe_time = net([memory], dims=dims)
            TEMP += exe_time
        shape_t.append(TEMP / 20)
    RES = np.multiply(np.array(shape_t), np.array(shape_num))
    RES = RES.tolist()
    FPS = 1000 / (sum(RES) / 7176)
    print("fps:", FPS)
    del net
    release_acl(DEVICE)
