#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.mask_rcnn_fpn import model
from ..common.train import train

from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import RegNet
from detectron2.modeling.backbone.regnet import SimpleStem, ResBottleneckBlock


# Replace default ResNet with RegNetX-4GF from the DDS paper. Config source:
# https://github.com/facebookresearch/pycls/blob/2c152a6e5d913e898cca4f0a758f41e6b976714d/configs/dds_baselines/regnetx/RegNetX-4.0GF_dds_8gpu.yaml#L4-L9  # noqa
model.backbone.bottom_up = L(RegNet)(
    stem_class=SimpleStem,
    stem_width=32,
    block_class=ResBottleneckBlock,
    depth=23,
    w_a=38.65,
    w_0=96,
    w_m=2.43,
    group_width=40,
    freeze_at=2,
    norm="FrozenBN",
    out_features=["s1", "s2", "s3", "s4"],
)
model.pixel_std = [57.375, 57.120, 58.395]

optimizer.weight_decay = 5e-5
train.init_checkpoint = (
    "https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906383/RegNetX-4.0GF_dds_8gpu.pyth"
)
# RegNets benefit from enabling cudnn benchmark mode
train.cudnn_benchmark = True
