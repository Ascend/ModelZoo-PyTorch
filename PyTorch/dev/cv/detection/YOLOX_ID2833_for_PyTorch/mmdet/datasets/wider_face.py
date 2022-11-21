
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

# Copyright (c) Open-MMLab. All rights reserved.    
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv

from .builder import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class WIDERFaceDataset(XMLDataset):
    """Reader for the WIDER Face dataset in PASCAL VOC format.

    Conversion scripts can be found in
    https://github.com/sovrasov/wider-face-pascal-voc-annotations
    """
    CLASSES = ('face', )

    PALETTE = [(0, 255, 0)]

    def __init__(self, **kwargs):
        super(WIDERFaceDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from WIDERFace XML style annotation file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = f'{img_id}.jpg'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            folder = root.find('folder').text
            data_infos.append(
                dict(
                    id=img_id,
                    filename=osp.join(folder, filename),
                    width=width,
                    height=height))

        return data_infos
