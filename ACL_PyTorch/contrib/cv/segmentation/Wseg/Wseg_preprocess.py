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
import sys
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as tf


class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def undo(self, imgarr):
        proc_img = imgarr.copy()

        proc_img[..., 0] = (self.std[0] * imgarr[..., 0] + self.mean[0]) * 255.
        proc_img[..., 1] = (self.std[1] * imgarr[..., 1] + self.mean[1]) * 255.
        proc_img[..., 2] = (self.std[2] * imgarr[..., 2] + self.mean[2]) * 255.

        return proc_img

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

def load_img_name_list(file_path, index=0):
    img_gt_name_list = open(file_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[index].strip('/') for img_gt_name in img_gt_name_list]
    return img_name_list

def pad(image,pad_size):
    w, h = image.size

    pad_mask = Image.new("L", image.size)
    pad_height = pad_size[0] - h
    pad_width = pad_size[1] - w

    assert pad_height >= 0 and pad_width >= 0

    pad_l = max(0, pad_width // 2)
    pad_r = max(0, pad_width - pad_l)
    pad_t = max(0, pad_height // 2)
    pad_b = max(0, pad_height - pad_t)

    image = F.pad(image, (pad_l, pad_t, pad_r, pad_b), fill=0, padding_mode="constant")
    pad_mask = F.pad(pad_mask, (pad_l, pad_t, pad_r, pad_b), fill=1, padding_mode="constant")

    return image, pad_mask

def get_batch_bin(img,imgname):

    pad_size = [1024, 1024]
    scales = [1, 0.5, 1.5, 2.0]
    batch_size = 8
    use_flips = True

    transform = tf.Compose([np.asarray,
                            Normalize()])

    for i in range(batch_size):

        sub_idx = i % batch_size
        scale = scales[sub_idx // (2 if use_flips else 1)]

        flip = use_flips and sub_idx % 2

        target_size = (int(round(img.size[0] * scale)),
                       int(round(img.size[1] * scale)))

        s_img = img.resize(target_size, resample=Image.CUBIC)

        if flip:
            s_img = F.hflip(s_img)
        im_msc, ignore = pad(s_img,pad_size)
        im_msc = transform(im_msc)
        ignore = np.array(ignore).astype(im_msc.dtype)[..., np.newaxis]
        im_msc = F.to_tensor(im_msc * (1 - ignore))

        imgnm = imgname + "_" + str(i)
        im_msc = np.array(im_msc,dtype= np.float32)

        im_msc.tofile(os.path.join(bin_path, imgnm + '.bin'))

def preprocess(file_path, voc12_root,bin_path):

    img_name_list = load_img_name_list(file_path)
    print(img_name_list)

    if not os.path.exists(bin_path):
        os.makedirs(bin_path)

    for i in range(len(img_name_list)):
        print("===> ",i)
        imgnm = img_name_list[i][33:-4]
        img = Image.open(os.path.join(voc12_root, img_name_list[i])).convert('RGB')

        get_batch_bin(img,imgnm)

if __name__ == "__main__":

    voc12_root_path = os.path.abspath(sys.argv[1])
    file_path = os.path.abspath(sys.argv[2])
    bin_path = os.path.abspath(sys.argv[3])

    preprocess(file_path,voc12_root_path, bin_path)
