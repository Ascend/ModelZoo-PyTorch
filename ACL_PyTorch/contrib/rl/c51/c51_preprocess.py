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

from DeepRL.deep_rl.agent.CategoricalDQN_agent import *
from DeepRL.deep_rl.network.network_heads import *
from DeepRL.deep_rl.network.network_bodies import *


class C51Agent(CategoricalDQNAgent):
    def __init__(self):
        config = Config()
        config.categorical_v_min = -10
        config.categorical_v_max = 10
        config.categorical_n_atoms = 51
        config.atoms = np.linspace(config.categorical_v_min,
                                   config.categorical_v_max, config.categorical_n_atoms)
        config.task_fn = lambda: Task('BreakoutNoFrameskip-v4')
        config.eval_env = config.task_fn()
        config.network_fn = lambda: CategoricalNet(config.action_dim, 51, NatureConvBody())
        config.state_normalizer = ImageNormalizer()
        self.config = config
        self.network = config.network_fn()
        self.atoms = tensor(config.atoms)
        self.num = 0

    def eval_step(self, state, states_file, action_file):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        torch.save(state, '{0}/{1}.pt'.format(states_file, self.num))
        prediction = self.network(state)
        q = (prediction['prob'] * self.atoms).sum(-1)
        action = to_np(q.argmax(-1))
        torch.save(action, '{0}/{1}.pt'.format(action_file, self.num))
        self.num += 1
        self.config.state_normalizer.unset_read_only()
        return action

    def eval_episode(self, num, states_file, action_file):
        env = self.config.eval_env
        state = env.reset()
        b = 0
        result = 0
        while b < num:
            action = self.eval_step(state, states_file, action_file)
            state, reward, done, info = env.step(action)
            result += reward
            b += 1


def load_model(agent, model_file, stats_file):
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
    agent.network.load_state_dict(state_dict)
    with open(stats_file, 'rb') as f:
        agent.config.state_normalizer.load_state_dict(pickle.load(f))
    return agent


if __name__ == "__main__":
    model_file = sys.argv[1]
    stats_file = sys.argv[2]
    states_file = sys.argv[3]
    action_file = sys.argv[4]
    num = int(sys.argv[5])
    mkdir(states_file)
    mkdir(action_file)
    agent = load_model(C51Agent(), model_file, stats_file)
    agent.eval_episode(num, states_file, action_file)
