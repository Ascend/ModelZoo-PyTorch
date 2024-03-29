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
# yapf: disable
# Inference Speed is tested on NVIDIA V100
hrnet = [
    dict(
        config='configs/hrnet/fcn_hr18s_512x512_160k_ade20k.py',
        checkpoint='fcn_hr18s_512x512_160k_ade20k_20200614_214413-870f65ac.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=33.0),
    ),
    dict(
        config='configs/hrnet/fcn_hr18s_512x1024_160k_cityscapes.py',
        checkpoint='fcn_hr18s_512x1024_160k_cityscapes_20200602_190901-4a0797ea.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=76.31),
    ),
    dict(
        config='configs/hrnet/fcn_hr48_512x512_160k_ade20k.py',
        checkpoint='fcn_hr48_512x512_160k_ade20k_20200614_214407-a52fc02c.pth',
        eval='mIoU',
        metric=dict(mIoU=42.02),
    ),
    dict(
        config='configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py',
        checkpoint='fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=80.65),
    ),
]
pspnet = [
    dict(
        config='configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py',
        checkpoint='pspnet_r50-d8_512x1024_80k_cityscapes_20200606_112131-2376f12b.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=78.55),
    ),
    dict(
        config='configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py',
        checkpoint='pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=79.76),
    ),
    dict(
        config='configs/pspnet/pspnet_r101-d8_512x512_160k_ade20k.py',
        checkpoint='pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=44.39),
    ),
    dict(
        config='configs/pspnet/pspnet_r50-d8_512x512_160k_ade20k.py',
        checkpoint='pspnet_r50-d8_512x512_160k_ade20k_20200615_184358-1890b0bd.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=42.48),
    ),
]
resnest = [
    dict(
        config='configs/resnest/pspnet_s101-d8_512x512_160k_ade20k.py',
        checkpoint='pspnet_s101-d8_512x512_160k_ade20k_20200807_145416-a6daa92a.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=45.44),
    ),
    dict(
        config='configs/resnest/pspnet_s101-d8_512x1024_80k_cityscapes.py',
        checkpoint='pspnet_s101-d8_512x1024_80k_cityscapes_20200807_140631-c75f3b99.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=78.57),
    ),
]
fastscnn = [
    dict(
        config='configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py',
        checkpoint='fast_scnn_8x4_160k_lr0.12_cityscapes-0cec9937.pth',
        eval='mIoU',
        metric=dict(mIoU=70.96),
    )
]
deeplabv3plus = [
    dict(
        config='configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py', # noqa
        checkpoint='deeplabv3plus_r101-d8_769x769_80k_cityscapes_20200607_000405-a7573d20.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=80.98),
    ),
    dict(
        config='configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py', # noqa
        checkpoint='deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=80.97),
    ),
    dict(
        config='configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py', # noqa
        checkpoint='deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=80.09),
    ),
    dict(
        config='configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py', # noqa
        checkpoint='deeplabv3plus_r50-d8_769x769_80k_cityscapes_20200606_210233-0e9dfdc4.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=79.83),
    ),
]
vit = [
    dict(
        config='configs/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k.py',
        checkpoint='upernet_vit-b16_ln_mln_512x512_160k_ade20k-f444c077.pth',
        eval='mIoU',
        metric=dict(mIoU=47.73),
    ),
    dict(
        config='configs/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k.py',
        checkpoint='upernet_deit-s16_ln_mln_512x512_160k_ade20k-c0cd652f.pth',
        eval='mIoU',
        metric=dict(mIoU=43.52),
    ),
]
fp16 = [
    dict(
        config='configs/deeplabv3plus/deeplabv3plus_r101-d8_fp16_512x1024_80k_cityscapes.py', # noqa
        checkpoint='deeplabv3plus_r101-d8_fp16_512x1024_80k_cityscapes_20200717_230920-f1104f4b.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=80.46),
    )
]
swin = [
    dict(
        config='configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py', # noqa
        checkpoint='upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth', # noqa
        eval='mIoU',
        metric=dict(mIoU=44.41),
    )
]
# yapf: enable
