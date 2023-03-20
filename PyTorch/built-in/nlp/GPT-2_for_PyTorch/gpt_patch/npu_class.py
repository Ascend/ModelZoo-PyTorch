#! -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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
import torch_npu


class DropOutTask:
    def __init__(self, shape, dtype, device, p):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.p = p
        self.request_count = 0
        self.mask_queue = []


class NpuDropout(torch.nn.Dropout):
    task_dict = {}
    dropout_stream = None

    def __init__(self, p):
        super().__init__(p)

    def forward(self, x):
        return NpuDropout.dropout_functional(x, self.p, True)

    @classmethod
    def dropout_functional(cls, x, p, training=True):
        if isinstance(x, torch.Tensor):
            shape = x.shape
            dtype = x.dtype
            device = x.device
            do_mask_flag = True
            return_obj = x
        elif isinstance(x, list):
            shape, dtype, device = x
            do_mask_flag = False
            return_obj = None
        else:
            raise RuntimeError("input type error!")

        if p == 0 or not training:
            return return_obj
        key = (shape, dtype, device, p)
        if key not in NpuDropout.task_dict:
            dropout_task = DropOutTask(shape, dtype, device, p)
            dropout_task.request_count += 1
            NpuDropout.task_dict[key] = dropout_task
            return return_obj
        elif not NpuDropout.task_dict[key].mask_queue:
            NpuDropout.task_dict[key].request_count += 1
            return return_obj
        else:
            mask, event = NpuDropout.task_dict[key].mask_queue.pop(0)
            if do_mask_flag:
                return torch.npu_dropout_do_mask(x, mask, p)[0]
            else:
                return mask

    @classmethod
    def generate_mask(cls):
        with torch.npu.stream(cls.dropout_stream):
            with torch.no_grad():
                for _, task in cls.task_dict.items():
                    if len(task.mask_queue) < task.request_count:
                        for j in range(task.request_count - len(task.mask_queue)):
                            mask = torch.npu_dropout_gen_mask(task.shape, p=task.p, dtype=task.dtype,
                                                              device=task.device)
                            event = None
                            task.mask_queue.append((mask, event))

    @classmethod
    def enable_dropout_ensemble(cls, model):
        if cls.dropout_stream is None:
            cls.dropout_stream = torch.npu.Stream()

        def wait_stream_hook_func():
            def hook_function(module, inputs):
                torch.npu.current_stream().wait_stream(cls.dropout_stream)

            return hook_function

        model.register_forward_pre_hook(wait_stream_hook_func())

        def mask_gen_hook_func():
            def hook_function(module, inputs, outputs):
                with torch.npu.stream(cls.dropout_stream):
                    with torch.no_grad():
                        for _, task in cls.task_dict.items():
                            if len(task.mask_queue) < task.request_count:
                                for j in range(task.request_count - len(task.mask_queue)):
                                    mask = torch.npu_dropout_gen_mask(task.shape, p=task.p, dtype=task.dtype,
                                                                      device=task.device)
                                    event = None
                                    task.mask_queue.append((mask, event))

            return hook_function

        model.register_forward_hook(mask_gen_hook_func())


class MatmulApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, mat2):
        # y = a * b^T
        ctx.save_for_backward(self, mat2)
        result = torch.matmul(self, mat2.transpose(-2, -1))
        return result
    @staticmethod
    def backward(ctx, grad):
        # da: grad * b
        # db: grad^T * a
        self, mat2 = ctx.saved_tensors
        self_grad = torch.matmul(grad, mat2)
        mat2_grad = torch.matmul(grad.transpose(-2, -1), self)
        return self_grad, mat2_grad