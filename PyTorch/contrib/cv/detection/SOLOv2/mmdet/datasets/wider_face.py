# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path as osp
import xml.etree.ElementTree as ET

import mmcv

from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class WIDERFaceDataset(XMLDataset):
    """
    Reader for the WIDER Face dataset in PASCAL VOC format.
    Conversion scripts can be found in
    https://github.com/sovrasov/wider-face-pascal-voc-annotations
    """
    CLASSES = ('face', )

    def __init__(self, **kwargs):
        super(WIDERFaceDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = '{}.jpg'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            folder = root.find('folder').text
            img_infos.append(
                dict(
                    id=img_id,
                    filename=osp.join(folder, filename),
                    width=width,
                    height=height))

        return img_infos
