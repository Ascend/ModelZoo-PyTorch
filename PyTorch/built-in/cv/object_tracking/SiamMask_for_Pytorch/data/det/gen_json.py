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
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os.path import join, isdir
from os import mkdir
import glob
import xml.etree.ElementTree as ET
import json

js = {}
VID_base_path = './ILSVRC2015'
ann_base_path = join(VID_base_path, 'Annotations/DET/train/')
sub_sets = ('ILSVRC2013_train', 'ILSVRC2013_train_extra0', 'ILSVRC2013_train_extra1', 'ILSVRC2013_train_extra2', 'ILSVRC2013_train_extra3', 'ILSVRC2013_train_extra4', 'ILSVRC2013_train_extra5', 'ILSVRC2013_train_extra6', 'ILSVRC2013_train_extra7', 'ILSVRC2013_train_extra8', 'ILSVRC2013_train_extra9', 'ILSVRC2013_train_extra10', 'ILSVRC2014_train_0000', 'ILSVRC2014_train_0001','ILSVRC2014_train_0002','ILSVRC2014_train_0003','ILSVRC2014_train_0004','ILSVRC2014_train_0005','ILSVRC2014_train_0006')
for sub_set in sub_sets:
    sub_set_base_path = join(ann_base_path, sub_set)

    if 'ILSVRC2013_train' == sub_set:
        xmls = sorted(glob.glob(join(sub_set_base_path, '*', '*.xml')))
    else:
        xmls = sorted(glob.glob(join(sub_set_base_path, '*.xml')))
    n_imgs = len(xmls)
    for f, xml in enumerate(xmls):
        print('subset: {} frame id: {:08d} / {:08d}'.format(sub_set, f, n_imgs))
        xmltree = ET.parse(xml)
        objects = xmltree.findall('object')

        video = join(sub_set, xml.split('/')[-1].split('.')[0])

        for id, object_iter in enumerate(objects):
            bndbox = object_iter.find('bndbox')
            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            frame = '%06d' % (0)
            obj = '%02d' % (id)
            if video not in js:
                js[video] = {}
            if obj not in js[video]:
                js[video][obj] = {}
            js[video][obj][frame] = bbox

json.dump(js, open('train.json', 'w'), indent=4, sort_keys=True)
