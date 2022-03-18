# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# Copyright 2020 Huawei Technologies Co., Ltd
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import torch
import math
from torch.optim.optimizer import Optimizer, required
from change_data_ptr import change_data_ptr


def combine_tensor(list_of_tensor, copy_back=True):
    total_numel = 0
    for tensor in list_of_tensor:
        total_numel += tensor.storage().size()
    combined_tensor = torch.randn(total_numel).npu().to(list_of_tensor[0].dtype)

    idx = 0
    if copy_back:
        for tensor in list_of_tensor:
            temp = tensor.clone()
            temp.copy_(tensor)
            change_data_ptr(tensor, combined_tensor, idx)
            temp_data = tensor.data
            temp_data.copy_(temp)
            idx += temp.storage().size()
    else:
        for tensor in list_of_tensor:
            change_data_ptr(tensor, combined_tensor, idx)
            idx += tensor.storage().size()
    return combined_tensor


def recombine_tensor(size, combined_tensor, index=0):
    temp_grad = torch.zeros(size).npu().to(combined_tensor.dtype)
    change_data_ptr(temp_grad, combined_tensor, index)
    return temp_grad


class CombinedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, combine_grad=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(CombinedAdam, self).__init__(params, defaults)

        self.combined = combine_grad
        self.init_combine = False
        self.first_init = True
        self.opt_level_O2_has_bn = False
        self.combined_grad = []
        self.combined_weight = []

    def __setstate__(self, state):
        super(CombinedAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def split_combined_tensors(self, input_combined_grad_1, input_combined_grad_2=None):
        if len(self.combined_weight) > 0:
            # has big tensor before, release storage
            for tensor in self.combined_weight:
                tensor = None
            self.first_init = False
            self.combined_grad = []
            self.combined_weight = []

        index_ops, index_bn = 0, 0
        for param_group in self.param_groups:
            size_ops, size_bn = 0, 0
            ord_param_list = []
            spe_param_list = []
            check_param_size = 0
            for param in param_group["params"]:
                if param.requires_grad and param.grad is not None:
                    temp_size = param.grad.storage().size()
                    check_param_size += param.storage().size()
                    if input_combined_grad_1.data_ptr() <= param.grad.data_ptr() < input_combined_grad_1.data_ptr() + input_combined_grad_1.numel() * input_combined_grad_1.element_size():
                        size_ops += temp_size
                        ord_param_list.append(param)
                    else:
                        size_bn += temp_size
                        spe_param_list.append(param)
            self.combined_grad.append(recombine_tensor(size_ops, input_combined_grad_1, index_ops))
            self.combined_weight.append(combine_tensor(ord_param_list, copy_back=True))

            index_ops += size_ops
            if input_combined_grad_2 is not None:
                self.combined_grad.append(recombine_tensor(size_bn, input_combined_grad_2, index_bn))
                self.combined_weight.append(combine_tensor(spe_param_list, copy_back=True))
                index_bn += size_bn

    def _init_combined(self):
        if not self.init_combine:
            if hasattr(self, "_amp_stash"):
                stash = self._amp_stash
                if hasattr(stash, "all_fp32_params"):
                    if len(stash.grads_list) == 0:
                        raise RuntimeError("When use CombinedAdam, Apex O1 need to use combine_grad=True module!")
                    self.split_combined_tensors(stash.grads_list[-1])
                    self.init_combine = True
                elif hasattr(stash, "all_fp32_from_fp16_params"):
                    if len(stash.grads_list) == 0:
                        raise RuntimeError("When use CombinedAdam, Apex O2 need to usecombine_grad=True module!")
                    if stash.grads_list[1] is not None:
                        if stash.grads_list[2] is None:
                            self.split_combined_tensors(stash.grads_list[1])
                        else:
                            self.split_combined_tensors(stash.grads_list[1], stash.grads_list[2])
                            self.opt_level_O2_has_bn = True
                    else:
                        raise RuntimeError("Inapproperiate network which only have batchnorm layers!")
                    self.init_combine = True
            else:
                for param_group in self.param_groups:
                    lst_grad = []
                    lst_weight = []
                    for param in param_group["params"]:
                        if param.requires_grad and param.grad is not None:
                            lst_grad.append(param.grad)
                            lst_weight.append(param)
                    if len(lst_grad) > 0:
                        self.combined_grad.append(combine_tensor(lst_grad, True))
                        self.combined_weight.append(combine_tensor(lst_weight, True))
                        self.combined_momentum.append(torch.zeros_like(self.combined_grad[-1]))
                        self.init_combine = True
            for idx, tensor in enumerate(self.combined_weight):
                state = self.state[tensor]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(self.combined_weight[idx])
                    state["exp_avg_sq"] = torch.zeros_like(self.combined_weight[idx])

    def step_combined(self, idx, state, group):
        combined_weight = self.combined_weight[idx]
        combined_grad = self.combined_grad[idx]
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

        amsgrad = group['amsgrad']
        if amsgrad:
            if state["step"] == 0:
                state["max_exp_avg_sq"] = torch.zeros_like(combined_weight)
        beta1, beta2 = group['betas']

        state["step"] += 1
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]

        exp_avg.mul_(beta1).add_(combined_grad, alpha=1-beta1)
        exp_avg_sq.mul_(beta2).addcmul_(combined_grad, combined_grad, value=1-beta2)

        if amsgrad:
            max_exp_avg_sq = state["max_exp_avg_sq"]
            max_exp_avg_sq = torch.max(max_exp_avg_sq, exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().add_(group["eps"])
        else:
            denom = exp_avg_sq.sqrt().add_(group["eps"])
        step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
        if group["weight_decay"] != 0:
            combined_weight.data.add_(-group["weight_decay"] * group["lr"], combined_weight.data)

        combined_weight.data.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None, enable=True):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        idx = 0
        for group in self.param_groups:
            if self.combined:
                self._init_combined()
                state = self.state[self.combined_weight[idx]]
                self.step_combined(idx, state, group)
                if self.opt_level_O2_has_bn:
                    idx += 1
                    state = self.state[self.combined_weight[idx]]
                    self.step_combined(idx, state, group)
            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    amsgrad = group['amsgrad']

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        max_exp_avg_sq = torch.max(max_exp_avg_sq, exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                    else:
                        denom = exp_avg_sq.sqrt().add_(group['eps'])

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                    if group['weight_decay'] != 0:
                        p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                    # p.data.addcdiv_(-step_size, exp_avg, denom)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
            idx += 1
        return loss