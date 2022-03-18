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

from deep_rl.agent.DQN_agent import *
from deep_rl.network.network_bodies import *


parser = argparse.ArgumentParser()
parser.add_argument('--pth-path', type=str)
parser.add_argument('--onnx-path', type=str)


class DQNAgent(DQNAgent):
    def __init__(self):
        config = Config()
        config.task_fn = lambda: Task('BreakoutNoFrameskip-v4')
        config.eval_env = config.task_fn()
        config.history_length = 4
        config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
        self.network = config.network_fn()

    def pth2onnx(self, input_file, output_file):
        state_dict = torch.load(input_file, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        self.network.eval()
        input_names = ["input"]
        output_names = ["output"]
        dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}
        dummy_input = torch.randn(16, 4, 84, 84)
        torch.onnx.export(self.network, dummy_input, output_file, input_names=input_names, dynamic_axes=dynamic_axes,
                          output_names=output_names, opset_version=11, verbose=True)


def pth2onnx(input_file, output_file):
    agent = DQNAgent()
    agent.pth2onnx(input_file, output_file)


if __name__ == "__main__":
    args = parser.parse_args()
    input_file = args.pth_path
    output_file = args.onnx_path
    pth2onnx(input_file, output_file)
