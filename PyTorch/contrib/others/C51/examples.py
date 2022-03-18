#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
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

from deep_rl import *


# DQN
def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    config.history_length = 1
    config.batch_size = 10
    config.discount = 0.99
    config.max_steps = 1e5

    replay_kwargs = dict(
        memory_size=int(1e4),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length)

    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.target_network_update_freq = 200
    config.exploration_steps = 1000
    # config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.async_actor = False
    run_steps(DQNAgent(config))


def dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('n_step', 1)
    kwargs.setdefault('replay_cls', UniformReplay)
    kwargs.setdefault('async_replay', True)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    # config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)
    config.batch_size = 32
    config.discount = 0.99
    config.history_length = 4
    config.max_steps = int(2e7)
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        n_step=config.n_step,
        discount=config.discount,
        history_length=config.history_length,
    )
    config.replay_fn = lambda: ReplayWrapper(config.replay_cls, replay_kwargs, config.async_replay)
    config.replay_eps = 0.01
    config.replay_alpha = 0.5
    config.replay_beta = LinearSchedule(0.4, 1.0, config.max_steps)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    # config.exploration_steps = 100
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.double_q = False
    config.async_actor = True
    run_steps(DQNAgent(config))


# C51
def categorical_dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms, FCBody(config.state_dim))
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)

    config.batch_size = 10
    replay_kwargs = dict(
        memory_size=int(1e4),
        batch_size=config.batch_size)
    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async_=True)

    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.categorical_v_max = 100
    config.categorical_v_min = -100
    config.categorical_n_atoms = 50
    config.gradient_clip = 5
    config.sgd_update_frequency = 4

    config.eval_interval = int(5e3)
    config.max_steps = 1e5
    run_steps(CategoricalDQNAgent(config))


def categorical_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms, NatureConvBody())
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    config.batch_size = 32
    replay_kwargs = dict(
        memory_size=int(1e6),
        batch_size=config.batch_size,
        history_length=4,
    )
    config.replay_fn = lambda: ReplayWrapper(UniformReplay, replay_kwargs, async_=False)

    config.discount = 0.99
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.categorical_v_max = 10
    config.categorical_v_min = -10
    config.categorical_n_atoms = 51
    config.sgd_update_frequency = 4
    config.gradient_clip = 0.5
    run_steps(CategoricalDQNAgent(config))
