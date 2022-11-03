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
import json

print('load json (raw ytb_vos info), please wait 10 seconds~')
ytb_vos = json.load(open('instances_train.json', 'r'))
snippets = dict()
for k, v in ytb_vos.items():
    video = dict()
    for i, o in enumerate(list(v)):
        obj = v[o]
        snippet = dict()
        trackid = "{:02d}".format(i)
        for frame in obj:
            file_name = frame['file_name']
            frame_name = '{:06d}'.format(int(file_name.split('/')[-1]))
            bbox = frame['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            snippet[frame_name] = bbox
        video[trackid] = snippet
    snippets['train/'+k] = video
        
train = snippets

json.dump(train, open('train.json', 'w'), indent=4, sort_keys=True)
print('done!')
