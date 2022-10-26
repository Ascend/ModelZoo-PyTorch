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


import torch.utils.data as data
import os
from PIL import Image
from utils import preprocess_image


class CelebASegmentation(data.Dataset):
  CLASSES = ['background' ,'skin','nose','eye_g','l_eye','r_eye','l_brow','r_brow','l_ear','r_ear','mouth','u_lip','l_lip','hair','hat','ear_r','neck_l','neck','cloth']

  def __init__(self, root, transform=None, crop_size=None):
    self.root = root
    self.transform = transform
    self.crop_size = crop_size

    self.images = []
    subdirs = next(os.walk(self.root))[1] #quick trick to get all subdirectories
    for subdir in subdirs:
        curr_images = [os.path.join(self.root,subdir,file) for file in os.listdir(os.path.join(self.root,subdir)) if file.endswith('.png')]
        self.images += curr_images


  def __getitem__(self, index):
    _img = Image.open(self.images[index]).convert('RGB')
    _img=_img.resize((513,513),Image.BILINEAR)
    _img = preprocess_image(_img,flip=False,scale=None,crop=(self.crop_size, self.crop_size))

    if self.transform is not None:
        _img = self.transform(_img)

    return _img

  def __len__(self):
    return len(self.images)
