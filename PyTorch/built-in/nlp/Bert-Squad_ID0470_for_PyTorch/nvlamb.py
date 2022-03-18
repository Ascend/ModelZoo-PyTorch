# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

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


class Nvlamb(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01,
                 amsgrad=False, adam_w_mode=True,
                 grad_averaging=True, set_grad_none=True,
                 max_grad_norm=1.0, use_nvlamb=False):
        if amsgrad:
            raise RuntimeError('Nvlamb does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        max_grad_norm=max_grad_norm)
        super(Nvlamb, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none
        self.use_nvlamb = use_nvlamb

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(Nvlamb, self).zero_grad()

    def multi_tensor_l2norm(self, tensor_list, ret_per_tensor=False):
        tensor_list_tmp = []
        if ret_per_tensor:
            for i in range(0, len(tensor_list)):
                tensor_list_tmp.append(torch.norm(tensor_list[i], 2))
            return tensor_list_tmp
        else:
            for i in range(0, len(tensor_list)):
                tensor_list_tmp.append(tensor_list[i].view(-1))
            tensor_cat = torch.cat(tensor_list_tmp, 0)
            tensor_norm = torch.norm(tensor_cat, 2)
            return tensor_norm

    def multi_tensor_lamb(self, tensor_list, lr, beta1, beta2, eps, step, bias_correction, weight_decay, grad_averaging,
                          adam_w_mode, global_grad_norm, max_grad_norm, use_nvlamb):
        def lamb_stage_1(tensor_list, beta1, beta2, beta3, beta1_correction, beta2_correction, eps, mode, weight_decay,
                         global_grad_norm, max_grad_norm):
            clipped_grad_norm = (global_grad_norm / max_grad_norm) if global_grad_norm > max_grad_norm else 1.0
            g, _, m, v = tensor_list
            if (weight_decay == 0):
                p = torch.zeros_like(tensor_list[1])
            else:
                p = tensor_list[1]
            if mode == 0:
                scale_grad = torch.div(g, clipped_grad_norm)
                scale_grad = scale_grad.add(p, alpha=weight_decay)
                m.mul_(beta1).add_(scale_grad, alpha=beta3)
                v.mul_(beta2).addcmul_(scale_grad, scale_grad, value=1 - beta2)
                next_m_unbiased = torch.div(m, beta1_correction)
                next_v_unbiased = torch.div(v, beta2_correction)
                denom = torch.sqrt(next_v_unbiased).add(eps)
                g.copy_(torch.div(next_m_unbiased, denom))
            else:
                scale_grad = torch.div(g, clipped_grad_norm)
                m.mul_(beta1).add_(scale_grad, alpha=beta3)
                v.mul_(beta2).addcmul_(scale_grad, scale_grad, value=1 - beta2)
                next_m_unbiased = torch.div(m, beta1_correction)
                next_v_unbiased = torch.div(v, beta2_correction)
                denom = torch.sqrt(next_v_unbiased).add(eps)
                g.copy_(torch.div(next_m_unbiased, denom).add(p, alpha=weight_decay))

        def lamb_stage_2(tensor_list, update_norm, param_norm, lr, weight_decay, use_nvlamb):
            ratio = lr
            update, p = tensor_list
            if use_nvlamb or weight_decay != 0:
                ratio = lr * (param_norm / update_norm) if (update_norm != 0 and param_norm != 0) else lr
            p.add_(update, alpha=-ratio)

        if len(tensor_list) != 4:
            raise RuntimeError("it should contain 4 tensor")
        beta1_correction = 1.0
        beta2_correction = 1.0
        if bias_correction == 1:
            beta1_correction = 1 - beta1 ** step
            beta2_correction = 1 - beta2 ** step
        beta3 = 1.0
        if grad_averaging == 1:
            beta3 = 1 - beta1
        g, p, m, v = tensor_list
        param_norm = self.multi_tensor_l2norm(tensor_list[1], True)
        for i in range(0, len(g)):
            tensor_list_single = [g[i], p[i], m[i], v[i]]
            lamb_stage_1(tensor_list_single, beta1, beta2, beta3, beta1_correction, beta2_correction, eps,
                         adam_w_mode, weight_decay, global_grad_norm, max_grad_norm)
        update_norm = self.multi_tensor_l2norm(g, True)
        for i in range(0, len(g)):
            lamb_stage_2([g[i], p[i]], update_norm[i], param_norm[i], lr, weight_decay, use_nvlamb)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
             closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        # create separate grad lists for fp32 and fp16 params
        g_all_32, g_all_16 = [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.dtype == torch.float32:
                    g_all_32.append(p.grad.data)
                elif p.dtype == torch.float16:
                    g_all_16.append(p.grad.data)
                else:
                    raise RuntimeError('Nvlamb only support fp16 and fp32.')
        device = self.param_groups[0]['params'][0].device
        g_norm_32 = torch.zeros(1, device=device)
        g_norm_16 = torch.zeros(1, device=device)
        # compute grad norm for two lists
        if len(g_all_32) > 0:
            g_norm_32 = self.multi_tensor_l2norm(g_all_32, False)

        if len(g_all_16) > 0:
            g_norm_16 = self.multi_tensor_l2norm(g_all_16, False)
        # blend two grad norms to get global grad norm
        global_grad_norm = self.multi_tensor_l2norm([g_norm_32, g_norm_16], False)
        max_grad_norm = self.defaults['max_grad_norm']

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('Nvlamb does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('Nvlamb only support fp16 and fp32.')
            if (len(g_16) > 0):
                self.multi_tensor_lamb(
                    [g_16, p_16, m_16, v_16],
                    group['lr'], beta1, beta2, group['eps'], group['step'],
                    bias_correction, group['weight_decay'], grad_averaging, self.adam_w_mode,
                    global_grad_norm, max_grad_norm, self.use_nvlamb)
            if (len(g_32) > 0):
                self.multi_tensor_lamb(
                    [g_32, p_32, m_32, v_32],
                    group['lr'], beta1, beta2, group['eps'], group['step'],
                    bias_correction, group['weight_decay'], grad_averaging, self.adam_w_mode,
                    global_grad_norm, max_grad_norm, self.use_nvlamb)
        return loss