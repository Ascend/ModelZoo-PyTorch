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

# Text Recognition Testing set, including:
# Regular Datasets: IIIT5K, SVT, IC13
# Irregular Datasets: IC15, SVTP, CT80

test_root = 'data/mixture'

test_img_prefix1 = f'{test_root}/IIIT5K/'
test_img_prefix2 = f'{test_root}/svt/'
test_img_prefix3 = f'{test_root}/icdar_2013/'
test_img_prefix4 = f'{test_root}/icdar_2015/'
test_img_prefix5 = f'{test_root}/svtp/'
test_img_prefix6 = f'{test_root}/ct80/'

test_ann_file1 = f'{test_root}/IIIT5K/test_label.txt'
test_ann_file2 = f'{test_root}/svt/test_label.txt'
test_ann_file3 = f'{test_root}/icdar_2013/test_label_1015.txt'
test_ann_file4 = f'{test_root}/icdar_2015/test_label.txt'
test_ann_file5 = f'{test_root}/svtp/test_label.txt'
test_ann_file6 = f'{test_root}/ct80/test_label.txt'

test1 = dict(
    type='OCRDataset',
    img_prefix=test_img_prefix1,
    ann_file=test_ann_file1,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

test2 = {key: value for key, value in test1.items()}
test2['img_prefix'] = test_img_prefix2
test2['ann_file'] = test_ann_file2

test3 = {key: value for key, value in test1.items()}
test3['img_prefix'] = test_img_prefix3
test3['ann_file'] = test_ann_file3

test4 = {key: value for key, value in test1.items()}
test4['img_prefix'] = test_img_prefix4
test4['ann_file'] = test_ann_file4

test5 = {key: value for key, value in test1.items()}
test5['img_prefix'] = test_img_prefix5
test5['ann_file'] = test_ann_file5

test6 = {key: value for key, value in test1.items()}
test6['img_prefix'] = test_img_prefix6
test6['ann_file'] = test_ann_file6

test_list = [test1]
