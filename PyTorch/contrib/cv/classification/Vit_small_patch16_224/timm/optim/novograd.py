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

import torch
from torch.optim.optimizer import Optimizer
import math


class NovoGrad(Optimizer):
    def __init__(self, params, grad_averaging=False, lr=0.1, betas=(0.95, 0.98), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(NovoGrad, self).__init__(params, defaults)
        self._lr = lr
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._eps = eps
        self._wd = weight_decay
        self._grad_averaging = grad_averaging

        self._momentum_initialized = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if not self._momentum_initialized:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('NovoGrad does not support sparse gradients')

                    v = torch.norm(grad)**2
                    m = grad/(torch.sqrt(v) + self._eps) + self._wd * p.data
                    state['step'] = 0
                    state['v'] = v
                    state['m'] = m
                    state['grad_ema'] = None
            self._momentum_initialized = True

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['step'] += 1

                step, v, m = state['step'], state['v'], state['m']
                grad_ema = state['grad_ema']

                grad = p.grad.data
                g2 = torch.norm(grad)**2
                grad_ema = g2 if grad_ema is None else grad_ema * \
                    self._beta2 + g2 * (1. - self._beta2)
                grad *= 1.0 / (torch.sqrt(grad_ema) + self._eps)

                if self._grad_averaging:
                    grad *= (1. - self._beta1)

                g2 = torch.norm(grad)**2
                v = self._beta2*v + (1. - self._beta2)*g2
                m = self._beta1*m + (grad / (torch.sqrt(v) + self._eps) + self._wd * p.data)
                bias_correction1 = 1 - self._beta1 ** step
                bias_correction2 = 1 - self._beta2 ** step
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                state['v'], state['m']  = v, m
                state['grad_ema'] = grad_ema
                p.data.add_(-step_size, m)
        return loss
