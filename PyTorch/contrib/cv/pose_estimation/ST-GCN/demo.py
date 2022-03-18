# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: utf-8 -*-
'''
demo.py
'''
import torch
import numpy as np
from apex import amp
import torch.distributed as dist
import torch.optim as optim
from collections import OrderedDict


def load_weights(model, weights_path):
    print('Load weights from {}.'.format(weights_path))
    weights = torch.load(weights_path)
    weights = OrderedDict([[k.split('module.')[-1],
                            v.cpu()] for k, v in weights.items()])
    try:
        model.load_state_dict(weights)
    except (KeyError, RuntimeError):
        state = model.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        for d in diff:
            print('Can not find weights [{}].'.format(d))
        state.update(weights)
        model.load_state_dict(state)
    return model


def build_model():
    from net.st_gcn import Model
    torch.npu.set_device('npu:0')
    model = Model(in_channels=3,
                  num_class=400,
                  edge_importance_weighting=True,
                  graph_args={'layout': "openpose",
                              'strategy': "spatial"})
    model = model.npu()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9)
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O2", loss_scale=1024)
    model = load_weights(
        model, './work_dir/recognition/kinetics_skeleton/ST_GCN/best_model_8p.pt')
    model.eval()
    return model


def get_raw_data():
    # The dataset that ST-GCN uses is generated by several steps.
    # First, process the source dataset, Kinetics, a dataset composed of Youtube videos.
    # Second, use OpenPose to generate person keypoints.
    # Finally, deal with the person keypoints.

    # The raw_data's source Youtube video ip address is https://www.youtube.com/watch?v=--6bJUbfpnQ
    # raw_data.npy is the corrsponding data of the above video.
    inp = np.load('tools/raw_data.npy')
    inputs = torch.from_numpy(inp).npu()
    return inputs


def pre_process(raw_data):
    return raw_data


def post_process(output_tensor):
    return torch.argmax(output_tensor, 1)


if __name__ == '__main__':
    raw_data = get_raw_data()

    model = build_model()

    input_tensor = pre_process(raw_data)

    output_tensor = model(input_tensor)

    result = post_process(output_tensor)

    print(result)
