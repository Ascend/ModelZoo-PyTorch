# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import glob
import time
from datetime import datetime

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import numpy as np

import gym

from PPO import PPO

def get_args_parser():
    parser = argparse.ArgumentParser(description = "Test Config")
    parser.add_argument("--env-name", type=str, default="RoboschoolWalker2d-v1", help="env name")
    parser.add_argument("--has-continuous-action-space", action="store_true", default=False, help="action space is continuous or discrete")
    parser.add_argument("--max-ep-len", type=int, default=1000, help="max timesteps in one episode")
    parser.add_argument("--action-std", type=float, default=0.1, help="set same std for action distribution which was used while saving")
    parser.add_argument('--render', action='store_true', default=False, help='render environment on screen')
    parser.add_argument('--frame-delay', type=int, default=0, help='add delay b/w frames')
    parser.add_argument('--total-test-episodes', type=int, default=10, help='total num of testing episodes')
    parser.add_argument("--K-epochs", type=int, default=80, help="update policy for K epochs")
    parser.add_argument("--eps-clip", type=float, default=0.2, help="clip parameter for PPO")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lr-actor", type=float, default=0.0003, help="learning rate for actor network")
    parser.add_argument("--lr-critic", type=float, default=0.001, help="learning rate for critic network")
    parser.add_argument("--random-seed", type=int, default=0, help="set random seed if required")
    parser.add_argument('--ckpt-path', type=str, default='', help='path of checkpoint')
    args = parser.parse_args()
    return args

#################################### Testing ###################################
def test():
    args = get_args_parser()
    ################## hyperparameters ##################

    env_name = args.env_name
    has_continuous_action_space = args.has_continuous_action_space
    max_ep_len = args.max_ep_len                        # max timesteps in one episode
    action_std = args.action_std                        # set same std for action distribution which was used while saving

    render = args.render                                # render environment on screen
    frame_delay = args.frame_delay                      # if required; add delay b/w frames

    total_test_episodes = args.total_test_episodes      # total num of testing episodes

    K_epochs = args.K_epochs                            # update policy for K epochs
    eps_clip = args.eps_clip                            # clip parameter for PPO
    gamma = args.gamma                                  # discount factor

    lr_actor = args.lr_actor                            # learning rate for actor
    lr_critic = args.lr_critic                          # learning rate for critic

    #####################################################

    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = args.random_seed
    env.seed(random_seed)

    checkpoint_path = args.ckpt_path
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    torch.npu.set_compile_mode(jit_compile=False)
    test()
