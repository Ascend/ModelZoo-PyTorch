"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer
from apex.contrib.combine_tensors import combine_npu


class NpuFusedAdamP(Optimizer):
    """Implements AdamP algorithm.

    Currently NPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--npu_float_status" ./``.

    This version of NPU fused AdamP implements 1 fusions.

      * A combine-tensor apply launch that batches the elementwise updates applied to all the model's parameters
        into one or a few kernel launches.

    :class:`apex.optimizers.NpuFusedAdamP` may be used as a drop-in replacement for ``torch.optim.AdamP``::

        opt = apex.optimizers.NpuFusedAdamP(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedAdam` should be used with Amp.  Currently, if you wish to use :class:`NpuFusedAdamP`
    with Amp, only ``opt_level O1 and O2`` can be choosed::

        opt = apex.optimizers.NpuFusedAdamP(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O2")
        ...
        opt.step()


    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamP variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional, default: 1e-3): learning rate
        betas (Tuple[float, float], optional, default: (0.9, 0.999)): coefficients used
            for computing running averages of gradient and its square
        eps (float, optional, default: 1e-8): term added to the denominator to improve
            numerical stability
        weight_decay (float, optional, default: 1e-2): weight decay coefficient
        amsgrad (boolean, optional, default: False): whether to use the AMSGrad variant of
            this algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        self.is_npu_fused_optimizer = True
        super(NpuFusedAdamP, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NpuFusedAdamP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        x_norm = x.norm(dim=1).add_(eps)
        y_norm = y.norm(dim=1).add_(eps)
        dot = (x * y).sum(dim=1)

        return dot.abs() / x_norm / y_norm

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p).size(1)):
                p_n = p / view_func(p).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def _init_param_state(self, p):
        state = self.state[p]
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        else:
            exp_avg_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avg_tmp.copy_(state['exp_avg'])
            state['exp_avg'] = exp_avg_tmp

            exp_avg_sq_tmp = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avg_sq_tmp.copy_(state['exp_avg_sq'])
            state['exp_avg_sq'] = exp_avg_sq_tmp

    def _combine_group_param_states(self, group_index):
        group = self.param_groups[group_index]
        stash = self._amp_stash
        group_params_list = stash.params_lists_indexed_by_group[group_index]

        combined_param_states = []
        for params in group_params_list:
            step_list = []
            exp_avg_list = []
            exp_avg_sq_list = []
            max_exp_avg_sq_list = []

            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('NpuFusedAdamP does not support sparse gradients, '
                                       'please consider SparseAdam instead')

                self._init_param_state(p)
                state = self.state[p]
                step_list.append(state['step'])
                exp_avg_list.append(state['exp_avg'])
                exp_avg_sq_list.append(state['exp_avg_sq'])

            combined_step = 0
            combined_exp_avg = None
            combined_exp_avg_sq = None
            combined_max_exp_avg_sq = None

            if len(exp_avg_list) > 0:
                combined_step = step_list[0]
                combined_exp_avg = combine_npu(exp_avg_list)
                combined_exp_avg_sq = combine_npu(exp_avg_sq_list)
                combined_max_exp_avg_sq = combine_npu(max_exp_avg_sq_list)

            combined_state = defaultdict(dict)
            combined_state['step'] = combined_step
            combined_state['exp_avg'] = combined_exp_avg
            combined_state['exp_avg_sq'] = combined_exp_avg_sq
            combined_state['max_exp_avg_sq'] = combined_max_exp_avg_sq
            combined_param_states.append(combined_state)
        stash.combined_param_states_indexed_by_group[group_index] = combined_param_states

    def _combine_param_states_by_group(self):
        stash = self._amp_stash
        if stash.param_states_are_combined_by_group:
            return

        stash.combined_param_states_indexed_by_group = []
        for _ in self.param_groups:
            stash.combined_param_states_indexed_by_group.append([])

        for i, _ in enumerate(self.param_groups):
            self._combine_group_param_states(i)
        stash.param_states_are_combined_by_group = True

    def _group_step(self, group_index):
        group = self.param_groups[group_index]
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad.data
            state = self.state[p]
            state['step'] += 1


        beta1, beta2 = group['betas']
        nesterov = group['nesterov']

        stash = self._amp_stash
        combined_group_params = stash.combined_params_indexed_by_group[group_index]
        combined_group_grads = stash.combined_grads_indexed_by_group[group_index]
        combined_group_param_states = stash.combined_param_states_indexed_by_group[group_index]

        for combined_param, combined_grad, combined_param_state in zip(combined_group_params,
                                                                       combined_group_grads,
                                                                       combined_group_param_states):
            if combined_param is None or combined_grad is None:
                continue

            exp_avg, exp_avg_sq = combined_param_state['exp_avg'], combined_param_state['exp_avg_sq']

            combined_param_state['step'] += 1
            bias_correction1 = 1 - beta1 ** combined_param_state['step']
            bias_correction2 = 1 - beta2 ** combined_param_state['step']

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(combined_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(combined_grad, combined_grad, value=1 - beta2)
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            step_size = group['lr'] / bias_correction1

            if nesterov:
                perturb = (beta1 * exp_avg + (1 - beta1) * combined_grad) / denom
            else:
                perturb = exp_avg / denom

            # Projection
            wd_ratio = 1
            if len(combined_param.shape) > 1:
                perturb, wd_ratio = self._projection(combined_param, combined_grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

            # Weight decay
            if group['weight_decay'] > 0:
                combined_param.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

            # Step
            combined_param.add_(-step_size, perturb)

    @torch.no_grad()
    def step(self, closure=None):
        if not hasattr(self, "_amp_stash"):
            raise RuntimeError('apex.optimizers.NpuFusedAdamP should be used with AMP.')

        self._check_already_combined_params_and_grads()
        # combine params and grads first
        self._combine_params_and_grads_by_group()
        # then combine param states
        self._combine_param_states_by_group()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, _ in enumerate(self.param_groups):
            self._group_step(i)

        return loss
