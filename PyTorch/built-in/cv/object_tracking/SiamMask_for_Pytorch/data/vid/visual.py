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
from os.path import join
from os import listdir
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET

visual = False
color_bar = np.random.randint(0, 255, (90, 3))

VID_base_path = './ILSVRC2015'
ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
img_base_path = join(VID_base_path, 'Data/VID/train/')
sub_sets = sorted({'a', 'b', 'c', 'd', 'e'})
for sub_set in sub_sets:
    sub_set_base_path = join(ann_base_path, sub_set)
    videos = sorted(listdir(sub_set_base_path))
    for vi, video in enumerate(videos):
        print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))

        video_base_path = join(sub_set_base_path, video)
        xmls = sorted(glob.glob(join(video_base_path, '*.xml')))
        for xml in xmls:
            f = dict()
            xmltree = ET.parse(xml)
            size = xmltree.findall('size')[0]
            frame_sz = [int(it.text) for it in size]
            objects = xmltree.findall('object')
            if visual:
                im = cv2.imread(xml.replace('xml', 'JPEG').replace('Annotations', 'Data'))
            for object_iter in objects:
                trackid = int(object_iter.find('trackid').text)
                bndbox = object_iter.find('bndbox')
                bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                if visual:
                    pt1 = (int(bbox[0]), int(bbox[1]))
                    pt2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(im, pt1, pt2, color_bar[trackid], 3)
            if visual:
                cv2.imshow('img', im)
                cv2.waitKey(1)

print('done!')
