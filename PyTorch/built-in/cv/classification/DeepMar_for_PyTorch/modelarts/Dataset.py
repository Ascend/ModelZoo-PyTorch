# Copyright(C) 2022. Huawei Technologies Co.,Ltd
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


import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pickle
import copy

class AttDataset(data.Dataset):
    """
    person attribute dataset interface
    """
    def __init__(
        self, 
        dataset,
        partition,
        split='train',
        partition_idx=0,
        transform=None,
        target_transform=None,
        **kwargs):
        self.read_path = kwargs['real_path']
        if os.path.exists( dataset ):
            file = open(dataset, 'rb')
            self.dataset = pickle.load(file)
        else:
            print (dataset + ' does not exist in dataset.')
            raise ValueError
        if os.path.exists( partition ):
            part = open(partition, 'rb')
            self.partition = pickle.load(part)
        else:
            print (partition + ' does not exist in dataset.')
            raise ValueError
        if split not in self.partition:
            print (split + ' does not exist in dataset.')
            raise ValueError
        
        if partition_idx > len(self.partition[split])-1:
            print ('partition_idx is out of range in partition.')
            raise ValueError

        self.transform = transform
        self.target_transform = target_transform

        # create image, label based on the selected partition and dataset split
        self.root_path = self.dataset['root']
        self.att_name = [self.dataset['att_name'][i] for i in self.dataset['selected_attribute']]
        self.image = []
        self.label = []
        for idx in self.partition[split][partition_idx]:
            self.image.append(self.dataset['image'][idx])
            label_tmp = np.array(self.dataset['att'][idx])[self.dataset['selected_attribute']].tolist()
            self.label.append(label_tmp)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the index of the target class
        """
        imgname, target = self.image[index], self.label[index]
        # load image and labels
        imgname = os.path.join(self.read_path, self.dataset['root'], imgname)
        img = Image.open(imgname)
        if self.transform is not None:
            img = self.transform( img )
        
        # default no transform
        target = np.array(target).astype(np.float32)
        target[target == 0] = -1
        target[target == 2] = 0
        if self.target_transform is not None:
            target = self.transform( target )

        return img, target

    # useless for personal batch sampler
    def __len__(self):
        return len(self.image)

