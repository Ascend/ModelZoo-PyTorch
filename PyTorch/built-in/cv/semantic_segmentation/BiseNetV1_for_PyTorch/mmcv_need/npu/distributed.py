# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright 2020 Huawei Technologies Co., Ltd
#
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
# --------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.parallel import MMDistributedDataParallel
from mmcv.device.scatter_gather import scatter_kwargs


class NPUDistributedDataParallel(MMDistributedDataParallel):
    """The DDP module supports DataContainer.

    NPUDDP has one difference from MMDDP which moves data to NPU with coping
    instead of scattering.
    """

    def to_kwargs(self, inputs, kwargs, device_id):
        # Use `self.to_kwargs` instead of `self.scatter` in pytorch1.8
        # to move all tensors to device_id
        return scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def forward(self, *inputs, **kwargs):
        # Since the scatter methhod is not supported on the NPU
        # and the DDP class is rewritten, when the forward of DDP
        # is used, the NPU will mask the scatter branch,
        # resulting in thhe input not being placed on the device side.
        # So, forward has been rewritten hhere primarily to circumvent
        # this situation that would cause the device misalignment.
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            return super().forward(*inputs[0], **kwargs[0])
        return super().forward(*inputs, **kwargs)
