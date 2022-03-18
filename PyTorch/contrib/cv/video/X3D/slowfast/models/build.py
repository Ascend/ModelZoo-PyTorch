# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


"""Model construction functions."""

import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    # if torch.cuda.is_available():
        # assert (
            # cfg.NUM_GPUS <= torch.cuda.device_count()
        # ), "Cannot use more GPU devices than available"
    # else:
        # assert (
            # cfg.NUM_GPUS == 0
        # ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)


    # 由于APEX要在 DDP之前初始化，所以把该部分拿出
    # if cfg.NUM_GPUS:
    #     if gpu_id is None:
    #         # Determine the GPU used by the current process
    #         cur_device = torch.cuda.current_device()
    #     else:
    #         cur_device = gpu_id
    #     # Transfer the model to the current GPU device
    #     model = model.cuda(device=cur_device)
    # # Use multi-process data parallel model in the multi-gpu setting
    # if cfg.NUM_GPUS > 1:
    #     # Make model replica operate on the current device
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         module=model, device_ids=[cur_device], output_device=cur_device
    #     )
    return model
