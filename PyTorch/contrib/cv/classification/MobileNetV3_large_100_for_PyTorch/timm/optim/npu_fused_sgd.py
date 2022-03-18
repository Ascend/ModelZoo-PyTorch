#!/usr/bin/env python
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



import torch
from torch.optim.optimizer import Optimizer, required
from collections import defaultdict
from contrib.combine_tensors import combine_npu

class NpuFusedSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Currently NPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--npu_float_status" ./``.

    This version of fused SGD implements 1 fusions.

      * A combine-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.NpuFusedSGD` may be used as a drop-in replacement for ``torch.optim.SGD``::

        opt = apex.optimizers.NpuFusedSGD(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedSGD` should be used with Amp.  Currently, if you wish to use :class:`NpuFusedSGD` with Amp,
    only ``opt_level O2`` can be choosed::

        opt = apex.optimizers.NpuFusedSGD(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O2")
        ...
        opt.step()

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.is_npu_fused_optimizer = True
        super(NpuFusedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NpuFusedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def _init_param_state(self, p, momentum_buffer_in_state_before, weight_decay):
        d_p = p.grad
        state = self.state[p]
        if 'momentum_buffer' not in state:
            momentum_buffer_in_state_before = False
            if weight_decay != 0:
                d_p = d_p.add(p, alpha=weight_decay)
            state['momentum_buffer'] = torch.clone(d_p).detach()
        else:
            temp = torch.clone(d_p).detach()
            temp.copy_(state['momentum_buffer'])
            state['momentum_buffer'] = temp

    def _combine_group_param_states(self, group_index, momentum_buffer_in_state_before):
        group = self.param_groups[group_index]
        stash = self._amp_stash
        group_params_list = stash.params_lists_indexed_by_group[group_index]
        
        weight_decay = group['weight_decay']
        momentum = group['momentum']

        combined_param_states = []
        for params in group_params_list:
            if momentum == 0:
                combined_state = defaultdict(dict)
                combined_state['momentum_buffer'] = None
                combined_param_states.append(combined_state)
                continue

            momentum_buffer_list = []
            for p in params:
                if p.grad is None:
                    continue

                self._init_param_state(p, momentum_buffer_in_state_before, weight_decay)
                state = self.state[p]
                momentum_buffer_list.append(state['momentum_buffer'])

            combined_momentum_buffer = None
            if len(momentum_buffer_list) > 0:
                combined_momentum_buffer = combine_npu(momentum_buffer_list)
            
            combined_state = defaultdict(dict)
            combined_state['momentum_buffer'] = combined_momentum_buffer
            combined_param_states.append(combined_state)
        stash.combined_param_states_indexed_by_group[group_index] = combined_param_states

    def _combine_param_states_by_group(self, momentum_buffer_in_state_before):
        momentum_buffer_in_state_before = True

        stash = self._amp_stash
        if stash.param_states_are_combined_by_group:
            return

        stash.combined_param_states_indexed_by_group = []
        for group in self.param_groups:
            stash.combined_param_states_indexed_by_group.append([])

        for i, group in enumerate(self.param_groups):
            self._combine_group_param_states(i, momentum_buffer_in_state_before)
        stash.param_states_are_combined_by_group = True

    def _group_step(self, group_index, momentum_buffer_in_state_before):
        group = self.param_groups[group_index]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
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
            
            if weight_decay != 0:
                combined_grad = combined_grad.add(combined_param, alpha=weight_decay)
            if momentum != 0:
                buf = combined_param_state['momentum_buffer']
                if momentum_buffer_in_state_before:
                    buf.mul_(momentum).add_(combined_grad, alpha=1 - dampening)

                if nesterov:
                    combined_grad = combined_grad.add(buf, alpha=momentum)
                else:
                    combined_grad = buf

            combined_param.add_(combined_grad, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not hasattr(self, "_amp_stash"):
            raise RuntimeError('apex.optimizers.NpuFusedSGD should be used with AMP.')

        momentum_buffer_in_state_before = True
        self._check_already_combined_params_and_grads()
        # combine params and grads first
        self._combine_params_and_grads_by_group()
        # then combine param states
        self._combine_param_states_by_group(momentum_buffer_in_state_before)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        stash = self._amp_stash
        for i, group in enumerate(self.param_groups):
            self._group_step(i, momentum_buffer_in_state_before)

        return loss
