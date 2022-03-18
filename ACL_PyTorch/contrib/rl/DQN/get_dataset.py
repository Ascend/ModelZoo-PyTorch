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


from deep_rl import *
from deep_rl.agent.BaseAgent import BaseAgent


parser = argparse.ArgumentParser()
parser.add_argument('--pth-path', type=str)
parser.add_argument('--state-path', type=str)
parser.add_argument('--num', type=int, default=1000)


def eval_step(state, weight_path):
    config = Config()
    config.task_fn = lambda: Task('BreakoutNoFrameskip-v4')
    config.eval_env = config.task_fn()
    config.history_length = 4
    model = VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    state = config.state_normalizer(state)
    network = model
    q = network(state)['q']
    action = to_np(q.argmax(-1))
    config.state_normalizer.unset_read_only()
    return action


def eval_episode(weight_path, output_file, num):
    game = 'BreakoutNoFrameskip-v4'
    task_fn = lambda: Task(game)
    env = task_fn()
    state = env.reset()
    a = 0
    while num:
        action = eval_step(state, weight_path)
        state, reward, done, info = env.step(action)
        torch.save(state, '{0}/state{1}.model'.format(output_file, a))
        num-=1
        a+=1


if __name__ == "__main__":
    args = parser.parse_args()
    input_file = args.pth_path
    output_file = args.state_path
    num = args.num
    eval_episode(input_file, output_file, num)


