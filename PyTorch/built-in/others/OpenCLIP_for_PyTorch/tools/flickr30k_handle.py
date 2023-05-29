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
import pandas as pd

data_path = os.getcwd()

test_data_num = 1000

annotations = pd.read_table('results_20130124.token', sep='\t', header=None,
                            names=['image', 'caption'])
images = []
captions = []

for image in annotations['image']:
    image_name = image.strip().split('#')[0]
    image_path = os.path.join(data_path, 'flickr30k-images', image_name)
    images.append(image_path)
captions = annotations['caption']

index_flag = len(images)-test_data_num
train_images = images[:index_flag]
train_captions = captions[:index_flag]
test_images = images[index_flag:]
test_captions = captions[index_flag:]

train_data = pd.DataFrame({'img_key': train_images, 'caption_key': train_captions},columns =['img_key','caption_key'])
train_data.to_csv('flickr30k_train.csv', index=False)
test_data = pd.DataFrame({'img_key': test_images, 'caption_key': test_captions},columns =['img_key','caption_key'])
test_data.to_csv('flickr30k_test.csv', index=False)