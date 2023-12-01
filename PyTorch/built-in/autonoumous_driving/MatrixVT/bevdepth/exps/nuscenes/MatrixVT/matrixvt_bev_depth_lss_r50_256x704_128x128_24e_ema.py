# Copyright (c) Megvii Inc. All rights reserved.
# coding=utf-8
# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import psutil
from functools import partial
# isort: skip_file
from bevdepth.exps.base_cli import run_cli
# Basic Experiment
from bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_ema import \
    BEVDepthLightningModel as BaseExp # noqa
from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
# new model
from bevdepth.models.matrixvt_det import MatrixVT_Det
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

class MatrixVT_Exp(BaseExp):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = MatrixVT_Det(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)
        self.data_use_cbgs = True

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch_npu.optim.NpuFusedAdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(
        MatrixVT_Exp,
        'matrixvt_bev_depth_lss_r50_256x704_128x128_24e_ema_cbgs',
        use_ema=True,
    )
