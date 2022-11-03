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
import json
import glob
import xml.etree.ElementTree as ET

VID_base_path = './ILSVRC2015'
ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
img_base_path = join(VID_base_path, 'Data/VID/train/')
sub_sets = sorted({'ILSVRC2015_VID_train_0000', 'ILSVRC2015_VID_train_0001', 'ILSVRC2015_VID_train_0002', 'ILSVRC2015_VID_train_0003', 'val'})

vid = []
for sub_set in sub_sets:
    sub_set_base_path = join(ann_base_path, sub_set)
    videos = sorted(listdir(sub_set_base_path))
    s = []
    for vi, video in enumerate(videos):
        print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))
        v = dict()
        v['base_path'] = join(sub_set, video)
        v['frame'] = []
        video_base_path = join(sub_set_base_path, video)
        xmls = sorted(glob.glob(join(video_base_path, '*.xml')))
        for xml in xmls:
            f = dict()
            xmltree = ET.parse(xml)
            size = xmltree.findall('size')[0]
            frame_sz = [int(it.text) for it in size]
            objects = xmltree.findall('object')
            objs = []
            for object_iter in objects:
                trackid = int(object_iter.find('trackid').text)
                name = (object_iter.find('name')).text
                bndbox = object_iter.find('bndbox')
                occluded = int(object_iter.find('occluded').text)
                o = dict()
                o['c'] = name
                o['bbox'] = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                             int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                o['trackid'] = trackid
                o['occ'] = occluded
                objs.append(o)
            f['frame_sz'] = frame_sz
            f['img_path'] = xml.split('/')[-1].replace('xml', 'JPEG')
            f['objs'] = objs
            v['frame'].append(f)
        s.append(v)
    vid.append(s)
print('save json (raw vid info), please wait 1 min~')
json.dump(vid, open('vid.json', 'w'), indent=4, sort_keys=True)
print('done!')
