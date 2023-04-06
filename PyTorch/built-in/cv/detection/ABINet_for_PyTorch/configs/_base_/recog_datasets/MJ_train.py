# Copyright 2022 Huawei Technologies Co., Ltd
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

# Text Recognition Training set, including:
# Synthetic Datasets: Syn90k

train_root = 'data/mixture/Syn90k'

train_img_prefix = f'{train_root}/mnt/ramdisk/max/90kDICT32px'
train_ann_file = f'{train_root}/label.lmdb'

train = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix,
    ann_file=train_ann_file,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

train_list = [train]