# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from torch import nn
import torch
if torch.__version__ >= '1.8':
    import torch_npu


class NpuSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, c):
        B, c1, c3 = input.shape
        ctx.input_shape = input.shape
        result = torch_npu.npu_indexing(input, [0, 0, 0],
                                    [B, c1, c3 + c], [1, 1, 1])
        return result
    @staticmethod
    def backward(ctx, grad):
        _, _, c = grad.shape
        input_shape = ctx.input_shape
        _, _, c3 = input_shape
        pads = (0, 0, 0, 0, 0, c3 - c)
        self_grad = torch_npu.npu_pad(grad, pads)
        return self_grad, None

npu_slice = NpuSlice.apply

class NpuSlice2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, c):
        B, c1, c3 = input.shape
        ctx.input_shape = input.shape
        result = torch_npu.npu_indexing(input, [0, 0, 0],
                                    [B, c1 + c, c3], [1, 1, 1])
        return result
    @staticmethod
    def backward(ctx, grad):
        _, c, _ = grad.shape
        input_shape = ctx.input_shape
        _, c1, _ = input_shape
        pads = (0, 0, 0, c1 - c, 0, 0)
        self_grad = torch_npu.npu_pad(grad, pads)
        return self_grad, None

npu_slice2 = NpuSlice2.apply


class FastGELU(nn.Module):
    """fast version of nn.GELU()"""

    def __init__(self):
        super(FastGELU, self).__init__()

    def forward(self, x):
        return torch_npu.fast_gelu(x)

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

    def __init__(self, p, module_name=None):
        super().__init__(p)
        self.module_name = module_name

    def forward(self, x):
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

        if self.p == 0:
            return return_obj
        key = (shape, dtype, device, self.p)
        if key not in NpuDropout.task_dict:
            dropout_task = DropOutTask(shape, dtype, device, self.p)
            dropout_task.request_count += 1
            NpuDropout.task_dict[key] = dropout_task
            return return_obj
        elif not NpuDropout.task_dict[key].mask_queue:
            NpuDropout.task_dict[key].request_count += 1
            return return_obj
        else:
            mask, event = NpuDropout.task_dict[key].mask_queue.pop(0)
            if do_mask_flag:
                return torch.npu_dropout_do_mask(x, mask, self.p)[0]
            else:
                return mask

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

class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            #x = x[:, :, : -self.remove]
            x = npu_slice(x, -self.remove)
        return x
