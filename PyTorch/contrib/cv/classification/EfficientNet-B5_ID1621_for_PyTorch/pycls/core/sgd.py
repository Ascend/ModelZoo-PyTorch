# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
from pycls.core.combine_tensors import combine_npu

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
        super(NpuFusedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NpuFusedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def combine_param_state_by_group(self, momentum_buffer_in_state_before):
        if not hasattr(self, "_amp_stash"):
            raise RuntimeError('apex.optimizers.NpuFusedSGD should be used with AMP.')

        momentum_buffer_in_state_before = True

        stash = self._amp_stash
        if stash.param_state_combined:
            return
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            if momentum == 0:
                state_combined = defaultdict(dict)
                state_combined['momentum_buffer'] = None
                stash.param_state_combined_list.append(state_combined)
                continue

            momentum_buffer_list = []
            for p in group['params']:
                if p.grad is None:
                    continue

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

                momentum_buffer_list.append(state['momentum_buffer'])

            momentum_buffer_combined = None
            if len(momentum_buffer_list) > 0:
                momentum_buffer_combined = combine_npu(momentum_buffer_list)
            
            state_combined = defaultdict(dict)
            state_combined['momentum_buffer'] = momentum_buffer_combined
            stash.param_state_combined_list.append(state_combined)

        stash.param_state_combined = True

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
        self._combine_params_and_grads_by_group()
        self.combine_param_state_by_group(momentum_buffer_in_state_before)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        stash = self._amp_stash

        for i, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            params_combined = stash.params_combined_list[i]
            grads_combined = stash.grads_combined_list[i]
            if params_combined is None or grads_combined is None:
                continue
           
            if weight_decay != 0:
                grads_combined = grads_combined.add(params_combined, alpha=weight_decay)
            if momentum != 0:
                param_state = stash.param_state_combined_list[i]
                buf = param_state['momentum_buffer']
                if momentum_buffer_in_state_before:
                    buf.mul_(momentum).add_(grads_combined, alpha=1 - dampening)

                if nesterov:
                    grads_combined = grads_combined.add(buf, alpha=momentum)
                else:
                    grads_combined = buf

            params_combined.add_(grads_combined, alpha=-group['lr'])

        return loss
