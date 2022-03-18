# Copyright 2021 Huawei Technologies Co., Ltd
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
import json

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video


class UAVVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """

    def __init__(self, name, root, video_dir, init_rect, img_names,
                 gt_rect, attr, load_img=False):
        super(UAVVideo, self).__init__(name, root, video_dir,
                                       init_rect, img_names, gt_rect, attr, load_img)


class UAVDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'UAV123', 'UAV20L'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """

    def __init__(self, name, dataset_root, load_img=False):
        super(UAVDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name + '.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading ' + name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = UAVVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'])

        # set attr
        attr = []
        for x in self.videos.values():
            attr += x.attr
        attr = set(attr)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)
