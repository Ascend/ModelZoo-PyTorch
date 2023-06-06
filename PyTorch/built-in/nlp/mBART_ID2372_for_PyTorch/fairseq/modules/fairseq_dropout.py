# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

def get_dropout_class():
    try:
        from torch import npu_dropout_do_mask
        return NpuFairseqDropout
    except:
        return FairseqDropout

class FairseqDropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.training or self.apply_during_inference:
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    "Cannot enable dropout during inference for module {} "
                    "because module_name was not set".format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    "Enabling dropout during inference for module: {}".format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info("Disabling dropout for module: {}".format(name))

class DropOutTask:
    def __init__(self, shape, dtype, device, p):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.p = p
        self.request_count = 0
        self.mask_queue = []

class NpuFairseqDropout(torch.nn.Dropout):
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
        if key not in NpuFairseqDropout.task_dict:
            dropout_task = DropOutTask(shape, dtype, device, self.p)
            dropout_task.request_count += 1
            NpuFairseqDropout.task_dict[key] = dropout_task
            return return_obj
        elif not NpuFairseqDropout.task_dict[key].mask_queue:
            NpuFairseqDropout.task_dict[key].request_count += 1
            return return_obj
        else:
            mask, event = NpuFairseqDropout.task_dict[key].mask_queue.pop(0)
            if do_mask_flag:
                return torch_npu.npu_dropout_do_mask(x, mask, self.p)[0]
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
                                    mask = torch_npu.npu_dropout_gen_mask(task.shape, p=task.p, dtype=task.dtype,
                                                                          device=task.device)
                                    event = None
                                    task.mask_queue.append((mask, event))
            return hook_function

        model.register_forward_hook(mask_gen_hook_func())
