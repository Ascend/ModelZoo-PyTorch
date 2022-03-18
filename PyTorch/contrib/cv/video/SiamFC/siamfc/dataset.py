# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import cv2
import os
import numpy as np
import pickle
import lmdb
import hashlib
from torch.utils.data.dataset import Dataset

from .config import config


class ImagnetVIDDataset(Dataset):
    def __init__(self, db, video_names, data_dir, z_transforms, x_transforms, training=True):
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms 
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]: x[1] for x in self.meta_data}     # x[0]:x[1] --> {'name1':seq1,'name2':seq2 }
        # filter traj len less than 2 , for example ILSVRC2015_train_00133002
        for key in self.meta_data.keys():  # .keys() : a dict_keys object, include all keys, i.e. ['name1', 'name2']
            trajs = self.meta_data[key]    # all image seq
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]

        self.txn = db.begin(write=False)
       
        self.num = len(self.video_names) if config.num_per_epoch is None or not training else config.num_per_epoch

    def imread(self, path):
        # lmdb is efficient
        key = hashlib.md5(path.encode()).digest()  # path -> key
        img_buffer = self.txn.get(key)  # key -> image
        img_buffer = np.frombuffer(img_buffer, np.uint8)  # get an array of 1D ndarray
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)  # get image from array

        # cv2.imread is slow
        # img=cv2.imread(path, cv2.IMREAD_COLOR)
        return img

    # suppose center(exemplar_id)=200, no exceeding total len, then low-idx=100, high_idx=300
    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'): 
        weights = list(range(low_idx, high_idx))  # list between 100-300
        weights.remove(center)  # remove exemplar_id(center_id)
        weights = np.array(weights)  # transfer to numpy
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def __getitem__(self, idx):

        idx = idx % len(self.video_names)
        video = self.video_names[idx]
        trajs = self.meta_data[video]
        # sample one trajs     np.random.choice refer to https://blog.csdn.net/ImwaterP/article/details/96282230
        trkid = np.random.choice(list(trajs.keys()))  # select one seq id
        traj = trajs[trkid]  # select the seq

        assert len(traj) > 1, "video_name: {}".format(video)  # judge if len>1

        # sample exemplar  
        exemplar_idx = np.random.choice(list(range(len(traj))))  # choose 1 id from the seq as the exemplar_idx
        exemplar_name = os.path.join(self.data_dir, video, traj[exemplar_idx]+".{:02d}.x.jpg".format(trkid))  # get path
        exemplar_img = self.imread(exemplar_name)
        exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)  # to rgb
        # sample instance, 0<= low_id < exemplar_idx < up_id <= exemplar_idx + 100
        low_idx = max(0, exemplar_idx - config.frame_range)
        up_idx = min(len(traj), exemplar_idx + config.frame_range)

        # create sample weight, if the sample are far away from center
        # the probability being choosen are high, config.sample_type='uniform' means same weight for all
        weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type) 
        instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx+1:up_idx], p=weights)
        instance_name = os.path.join(self.data_dir, video, instance+".{:02d}.x.jpg".format(trkid))
        instance_img = self.imread(instance_name)
        instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
        if np.random.rand(1) < config.gray_ratio:
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)
        exemplar_img = self.z_transforms(exemplar_img)  # 127x127
        instance_img = self.x_transforms(instance_img)  # (255-8x2) * (255-8x2)
        return exemplar_img, instance_img

    def __len__(self):
        return self.num
