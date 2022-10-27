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

import argparse
import torch

from DeepRL.deep_rl.agent.CategoricalDQN_agent import *
from DeepRL.deep_rl.network.network_utils import *
from DeepRL.deep_rl.network.network_bodies import *


parser = argparse.ArgumentParser("C51")
parser.add_argument('--model-path', type=str,  default='c51.model')
parser.add_argument('--onnx-path', type=str, default='c51.onnx')

class C51Net(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(C51Net, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        return pre_prob
    

class C51Agent(CategoricalDQNAgent):
    def __init__(self):
        config = Config()
        config.task_fn = lambda: Task('BreakoutNoFrameskip-v4')
        config.eval_env = config.task_fn()
        config.network_fn = lambda: C51Net(config.action_dim, 51, NatureConvBody())
        self.network = config.network_fn()

    def pth2onnx(self, input_file, output_file):
        state_dict = torch.load(input_file, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        dummy_input = torch.randn(1, 4, 84, 84)
        torch.onnx.export(self.network, dummy_input, output_file, export_params=True,input_names=['input'], output_names=['output'])


def pth2onnx(input_file, output_file):
    agent = C51Agent()
    agent.pth2onnx(input_file, output_file)


if __name__ == "__main__":
    args = parser.parse_args()
    input_file = args.model_path
    output_file = args.onnx_path
    pth2onnx(input_file, output_file)
