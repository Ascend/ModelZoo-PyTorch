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

import numpy as np

import torch
import torch.utils.data as data
import torchbiomed.utils as utils
from glob import glob
import os
import os.path
import SimpleITK as sitk

MIN_BOUND = -1000
MAX_BOUND = 400

image_dict = {}
label_dict = {}
mask_dict = {}
stats_dict = {}
test_split = []
train_split = []


def train_test_split(full, positive, test_fraction):
    negative = full - positive
    test_neg_count = int(np.ceil(len(negative)*test_fraction))
    test_pos_count = int(np.ceil(len(positive)*test_fraction))
    negative_list = list(negative)
    positive_list = list(positive)
    np.random.shuffle(positive_list)
    np.random.shuffle(negative_list)
    test_positive = set()
    for i in range(test_pos_count):
        test_positive |= set([positive_list[i]])
    train_positive = positive - test_positive
    if test_neg_count > 1:
        test_negative = set()
        for i in range(test_neg_count):
            test_negative |= set([negative_list[i]])
        train_negative = negative - test_negative
        train = list(train_positive | train_negative)
        test = list(test_positive | test_negative)
    else:
        train = list(train_positive)
        test = list(test_positive)
    np.random.shuffle(train)
    np.random.shuffle(test)
    return (train, test)


def load_image(root, series):
    if series in image_dict.keys():
        return image_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    # z, y, x = np.shape(img)
    # img = img.reshape((1, z, y, x))
    image_dict[series] = utils.truncate(img, MIN_BOUND, MAX_BOUND)
    stats_dict[series] = itk_img.GetOrigin(), itk_img.GetSpacing()
    return img


def load_label(root, series):
    if series in label_dict.keys():
        return label_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    if np.max(img) > 3400:
        img[img <= 3480] = 0
        img[img > 3480] = 1
    else:
        img[img != 0] = 1
    label_dict[series] = img.astype(np.uint8)
    return img


def load_mask(root, series):
    if series in mask_dict.keys():
        return mask_dict[series]
    img_file = os.path.join(root, series + ".mhd")
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)
    img[img != 0] = 1
    mask_dict[series] = img
    return img

def full_dataset(dir, images):
    image_path = os.path.join(dir, images)
    image_files = glob(os.path.join(image_path, "*.mhd"))
    image_list = []
    for name in image_files:
        image_list.append(os.path.basename(name)[:-4])
    return image_list

def make_dataset(dir, images, targets, seed, train, class_balance, partition, nonempty, test_fraction, mode):
    global image_dict, label_dict, test_split, train_split
    zero_tensor = None

    train = mode == "train"
    label_path = os.path.join(dir, targets)
    label_files = glob(os.path.join(label_path, "*.mhd"))
    label_list = []
    for name in label_files:
        label_list.append(os.path.basename(name)[:-4])

    sample_label = load_label(label_path, label_list[0])
    shape = np.shape(sample_label)
    if len(test_split) == 0 and os.path.isfile('torchbiomed/datasets/train_uids.txt') and os.path.isfile('torchbiomed/datasets/test_uids.txt'):
        # print('Load split from txt')
        train_split=open('torchbiomed/datasets/train_uids.txt').read().split('\n')[:-1]
        test_split=open('torchbiomed/datasets/test_uids.txt').read().split('\n')[:-1]
        # print('Get %d train cases and %d test cases'%(len(train_split),len(test_split)))
    if len(test_split) == 0:
        zero_tensor = np.zeros(shape, dtype=np.uint8)
        image_list = []
        image_path = os.path.join(dir, images)
        file_list=glob(image_path + "/*.mhd")
        for img_file in file_list:
            series = os.path.basename(img_file)[:-4]
            if series not in label_list:
                continue
            image_list.append(series)
            if series not in label_list:
                label_dict[series] = zero_tensor
        np.random.seed(seed)
        full = set(image_list)
        positives = set(label_list) & full
        train_split, test_split = train_test_split(full, positives, test_fraction)
    if train:
        keys = train_split
    else:
        keys = test_split
    part_list = []
    z, y, x = shape
    if partition is not None:
        z_p, y_p, x_p = partition
        z, y, x = shape
        z_incr, y_incr, x_incr = z // z_p, y // y_p, x // x_p
        assert z % z_p == 0
        assert y % y_p == 0
        assert x % x_p == 0
        for zi in range(z_p):
            zstart = zi*z_incr
            zend = zstart + z_incr
            for yi in range(y_p):
                ystart = yi*y_incr
                yend = ystart + y_incr
                for xi in range(x_p):
                    xstart = xi*x_incr
                    xend = xstart + x_incr
                    part_list.append(((zstart, zend), (ystart, yend), (xstart, xend)))
    else:
        part_list = [((0, z), (0, y), (0, x))]
    result = []
    target_means = []
    for key in keys:
        for part in part_list:
            target = load_label(label_path, key)
            if nonempty:
                if np.sum(utils.get_subvolume(target, part)) == 0:
                    continue
            target_means.append(np.mean(target))
            result.append((key, part))

    target_mean = np.mean(target_means)
    return (result, target_mean)

class LUNA16(data.Dataset):
    def __init__(self, root='.', images=None, targets=None, transform=None,
                 target_transform=None, co_transform=None, mode="train", seed=1,
                 class_balance=False, split=None, masks=None, nonempty=True,
                 test_fraction=0.15):
        if images is None:
            raise(RuntimeError("images must be set"))
        if targets is None and mode != "infer":
            raise(RuntimeError("both images and targets must be set if mode is not 'infer'"))
        if mode == "infer":
            imgs = full_dataset(root, images)
        else:
            imgs, target_mean = make_dataset(root, images, targets, seed, mode, class_balance, split, nonempty, test_fraction, mode)
            self.data_mean = target_mean
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images: " + os.path.join(root, images) + "\n"))

        self.mode = mode
        self.root = root
        self.imgs = imgs
        self.masks = None
        self.split = split
        if masks is not None:
            self.masks = os.path.join(self.root, masks)
        if targets is not None:
            self.targets = os.path.join(self.root, targets)
        self.images = os.path.join(self.root, images)
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

    def target_mean(self):
        return self.data_mean

    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "test":
            return self.__getitem_dev(index)
        elif self.mode == "infer":
            return self.__getitem_prod(index)

    def __getitem_prod(self, index):
        series = self.imgs[index]
        image = load_image(self.images, series)
        origin, spacing = stats_dict[series]
        image = image.astype(np.float32)
        if self.split is not None:
            batches = utils.partition_image(image, self.split)
        else:
            batches = [image]
        if self.transform is not None:
            batches = map(self.transform, batches)
            batches = [*batches]
        batches = torch.cat(batches)
        return batches, series, origin, spacing

    def __getitem_dev(self, index):
        series, bounds = self.imgs[index]
        (zs, ze), (ys, ye), (xs, xe) = bounds
        target = load_label(self.targets, series)
        target = target[zs:ze, ys:ye, xs:xe]
        target = torch.from_numpy(target.astype(np.int64))
        image = load_image(self.images, series)
        image = image[zs:ze, ys:ye, xs:xe]
        # optionally mask out lung volume from image
        if self.masks is not None:
            mask = load_mask(self.masks, series)
            mask = mask[zs:ze, ys:ye, xs:xe]
            image -= MIN_BOUND
            image = np.multiply(mask, image)
            image += MIN_BOUND
        image = image.transpose([1,2,0])
        img = image.astype(np.float32)

        if self.transform is not None:
            # print(img.shape, img.mean())
            img = self.transform(img)
            # print(img.shape, img.mean())
            img = img.unsqueeze(0)
            # print(img.shape)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.co_transform is not None:
            img, target = self.co_transform(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)
