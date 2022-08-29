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
"""Rollout Storage for PPO"""

import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


class RolloutStorage(object):
    """Rollout Storage for NAS. Using policy gradient without value predictions"""

    def __init__(self, num_steps, num_processes, action_size):
        """

        Args:
          num_steps: rollout length
          num_processes: samples per step
          action_size: flattened segmenter configuration

        """
        self.action_log_probs = np.zeros((num_steps * num_processes, 1))
        self.actions = np.zeros((num_steps * num_processes, action_size), dtype=np.int)
        self.rewards = np.zeros((num_steps * num_processes, 1))
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0

    def insert(self, action, log_prob, reward):
        inds = range(
            self.step * self.num_processes, (self.step + 1) * self.num_processes
        )
        self.actions[inds] = action
        self.action_log_probs[inds] = log_prob.item()
        self.rewards[inds] = reward

        self.step = (self.step + 1) % self.num_steps

    def generator(self, advantages, num_mini_batch):
        """Create a data loader from rollout buffer

        Args:
          advantages ([float]): list of advantage values of the rollout buffer
          num_mini_batch (int): number of batches to split the buffer into

        Returns:
          actions_batch: batch of actions
          rewards_batch: batch of rewards
          old_actions_log_probs_batch: batch of action log probabilities
          adv_targ: batch of estimated advantages

        """
        num_steps, num_processes = self.rewards.shape[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False
        )
        for indices in sampler:
            actions_batch = self.actions[indices]
            old_actions_log_probs_batch = self.action_log_probs[indices]
            actions_batch = self.actions[indices]
            rewards_batch = self.rewards[indices]
            adv_targ = advantages[indices]

            yield actions_batch, rewards_batch, old_actions_log_probs_batch, adv_targ
