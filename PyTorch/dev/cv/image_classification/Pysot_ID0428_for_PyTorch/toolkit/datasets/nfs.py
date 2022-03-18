#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#

import json
import os
import numpy as np

from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video


class NFSVideo(Video):
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
        super(NFSVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)

    # def load_tracker(self, path, tracker_names=None):
    #     """
    #     Args:
    #         path(str): path to result
    #         tracker_name(list): name of tracker
    #     """
    #     if not tracker_names:
    #         tracker_names = [x.split('/')[-1] for x in glob(path)
    #                 if os.path.isdir(x)]
    #     if isinstance(tracker_names, str):
    #         tracker_names = [tracker_names]
    #     # self.pred_trajs = {}
    #     for name in tracker_names:
    #         traj_file = os.path.join(path, name, self.name+'.txt')
    #         if os.path.exists(traj_file):
    #             with open(traj_file, 'r') as f :
    #                 self.pred_trajs[name] = [list(map(float, x.strip().split(',')))
    #                         for x in f.readlines()]
    #             if len(self.pred_trajs[name]) != len(self.gt_traj):
    #                 print(name, len(self.pred_trajs[name]), len(self.gt_traj), self.name)
    #         else:

    #     self.tracker_names = list(self.pred_trajs.keys())

class NFSDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "NFS30" or "NFS240"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(NFSDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = NFSVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          None)

        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
