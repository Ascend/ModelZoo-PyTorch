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
import os.path as osp

import mmcv

from mmseg.apis import inference_segmentor, init_segmentor


def test_test_time_augmentation_on_cpu():
    config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
    config = mmcv.Config.fromfile(config_file)

    # Remove pretrain model download for testing
    config.model.pretrained = None
    # Replace SyncBN with BN to inference on CPU
    norm_cfg = dict(type='BN', requires_grad=True)
    config.model.backbone.norm_cfg = norm_cfg
    config.model.decode_head.norm_cfg = norm_cfg
    config.model.auxiliary_head.norm_cfg = norm_cfg

    # Enable test time augmentation
    config.data.test.pipeline[1].flip = True

    checkpoint_file = None
    model = init_segmentor(config, checkpoint_file, device='cpu')

    img = mmcv.imread(
        osp.join(osp.dirname(__file__), 'data/color.jpg'), 'color')
    result = inference_segmentor(model, img)
    assert result[0].shape == (288, 512)
