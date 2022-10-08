# Copyright 2022 Huawei Technologies Co., Ltd.
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
# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmdet.datasets import DATASETS


def test_xml_dataset():
    dataconfig = {
        'ann_file': 'data/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
        'img_prefix': 'data/VOCdevkit/VOC2007/',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }]
    }
    XMLDataset = DATASETS.get('XMLDataset')

    class XMLDatasetSubClass(XMLDataset):
        CLASSES = None

    # get_ann_info and _filter_imgs of XMLDataset
    # would use self.CLASSES, we added CLASSES not NONE
    with pytest.raises(AssertionError):
        XMLDatasetSubClass(**dataconfig)
