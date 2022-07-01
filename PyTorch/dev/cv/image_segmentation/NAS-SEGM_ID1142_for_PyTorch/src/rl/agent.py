#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
"""Reinforcement Learning-based agent"""

from .gradient_estimators import REINFORCE, PPO
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


def create_agent(
    enc_num_layers,
    num_ops,
    num_agg_ops,
    lstm_hidden_size,
    lstm_num_layers,
    dec_num_cells,
    cell_num_layers,
    cell_max_repeat,
    cell_max_stride,
    ctrl_lr,
    ctrl_baseline_decay,
    ctrl_agent,
    ctrl_version="cvpr",
):
    """Create Agent

    Args:
      enc_num_layers (int) : size of initial sampling pool, number of encoder outputs
      num_ops (int) : number of unique operations
      num_agg_ops (int) : number of unique aggregation operations
      lstm_hidden_size (int) : number of neurons in RNN's hidden layer
      lstm_num_layers (int) : number of LSTM layers
      dec_num_cells (int) : number of cells in the decoder
      cell_num_layers (int) : number of layers in a cell
      cell_max_repeat (int) : maximum number of repeats the cell (template) can be repeated.
                              only valid for the 'wacv' controller
      cell_max_stride (int) : max stride of the cell (template). only for 'wacv'
      ctrl_lr (float) : controller's learning rate
      ctrl_baseline_decay (float) : controller's baseline's decay
      ctrl_agent (str) : type of agent's controller
      ctrl_version (str, either 'cvpr' or 'wacv') : type of microcontroller

    Returns:
      controller net that provides the sample() method
      gradient estimator

    """
    if ctrl_version == "cvpr":
        from rl.micro_controllers import MicroController as Controller
    elif ctrl_version == "wacv":
        from rl.micro_controllers import TemplateController as Controller

    controller = Controller(
        enc_num_layers=enc_num_layers,
        num_ops=num_ops,
        num_agg_ops=num_agg_ops,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        dec_num_cells=dec_num_cells,
        cell_num_layers=cell_num_layers,
        cell_max_repeat=cell_max_repeat,
        cell_max_stride=cell_max_stride,
    )
    if ctrl_agent == "ppo":
        agent = PPO(
            controller,
            clip_param=0.1,
            lr=ctrl_lr,
            baseline_decay=ctrl_baseline_decay,
            action_size=controller.action_size(),
        )
    elif ctrl_agent == "reinforce":
        agent = REINFORCE(controller, lr=ctrl_lr, baseline_decay=ctrl_baseline_decay)
    return agent


def train_agent(agent, sample):
    """Training controller"""
    config, reward, entropy, log_prob = sample
    action = agent.controller.config2action(config)
    loss, dist_entropy = agent.update((reward, action, log_prob))
