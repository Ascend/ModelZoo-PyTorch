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
import os
import cv2
import random
import numpy as np
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')


# Ref.: https://github.com/yjn870/ESPCN-pytorch/blob/ab84ee1bccb2978f2f9b88f9e0315d9be12c099e/prepare.py#L16
# Train Dataset Class
class SRTrainDataset(IterableDataset):
    def __init__(self, dirpath_images, scaling_factor, patch_size, stride):
        """ Training Dataset

        :param dirpath_images: path to training images directory
        :param scaling_factor: up-scaling factor to use, default = 3
        :param patch_size: size of sub-images to be extracted
        :param stride: stride used for extracting sub-images
        """

        self.dirpath_images = dirpath_images
        self.scaling_factor = scaling_factor
        self.patch_size = patch_size
        self.stride = stride

    def __iter__(self):
        for fpath_image in glob(os.path.join(self.dirpath_images, "*.png")):
            # Load HR image: rH x rW x C, r: scaling factor
            hr_image = Image.open(fpath_image).convert('RGB')
            hr_width = (hr_image.width // self.scaling_factor) * self.scaling_factor
            hr_height = (hr_image.height // self.scaling_factor) * self.scaling_factor
            hr_image = hr_image.resize((hr_width, hr_height), resample=Image.BICUBIC)

            # LR Image: H x W x C
            # As in paper, Sec. 3.2: sub-sample images by up-scaling factor
            lr_image = hr_image.resize((hr_width // self.scaling_factor, hr_height // self.scaling_factor), resample=Image.BICUBIC)

            hr_image = np.array(hr_image).astype(np.float32)
            lr_image = np.array(lr_image).astype(np.float32)

            # Convert BGR to YCbCr
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)

            # As per paper, using only the luminescence channel gave the best outcome
            hr_y = hr_image[:, :, 0]
            lr_y = lr_image[:, :, 0]

            # Get sub-image from Ihr and Ilr as per Sec. 3.2 in paper
            # using patch_size = 17 and stride = 13
            # This ensures that all pixels in the original image appear once and only once as the ground truth of the
            # training data
            rows = lr_y.shape[0]
            cols = lr_y.shape[1]
            for i in range(0, rows - self.patch_size + 1, self.stride):
                for j in range(0, cols - self.patch_size + 1, self.stride):
                    # lr_crop: w = 17, h = 17
                    lr_crop = lr_y[i:i + self.patch_size, j:j + self.patch_size]
                    # hr_crop: w = 17 * r, h = 17 * r
                    hr_crop = hr_y[i * self.scaling_factor:i * self.scaling_factor + self.patch_size * self.scaling_factor,
                              j * self.scaling_factor:j * self.scaling_factor + self.patch_size * self.scaling_factor]
                    lr_crop = np.expand_dims(lr_crop / 255.0, axis=0)
                    hr_crop = np.expand_dims(hr_crop / 255.0, axis=0)
                    yield lr_crop, hr_crop

    def __len__(self):
        return len(self.all_images)


# Valid Dataset Class
class SRValidDataset(IterableDataset):
    def __init__(self, dirpath_images, scaling_factor):
        """ Validation Dataset

        :param dirpath_images: path to validation images directory
        :param scaling_factor: up-scaling factor to use, default = 3
        """

        self.dirpath_images = dirpath_images
        self.scaling_factor = scaling_factor

    def __iter__(self):
        for fpath_image in glob(os.path.join(self.dirpath_images, "*.png")):
            # Load HR image: rH x rW x C, r: scaling factor
            hr_image = Image.open(fpath_image).convert('RGB')
            hr_width = (hr_image.width // self.scaling_factor) * self.scaling_factor
            hr_height = (hr_image.height // self.scaling_factor) * self.scaling_factor
            hr_image = hr_image.resize((hr_width, hr_height), resample=Image.BICUBIC)

            # LR Image: H x W x C
            # As in paper, Sec. 3.2: sub-sample images by up-scaling factor
            lr_image = hr_image.resize((hr_width // self.scaling_factor, hr_height // self.scaling_factor), resample=Image.BICUBIC)

            hr_image = np.array(hr_image).astype(np.float32)
            lr_image = np.array(lr_image).astype(np.float32)

            # Convert BGR to YCbCr
            hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
            lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)

            # As per paper, using only the luminescence channel gave the best outcome
            hr_y = hr_image[:, :, 0]
            lr_y = lr_image[:, :, 0]

            lr_y = np.expand_dims(lr_y / 255.0, axis=0)
            hr_y = np.expand_dims(hr_y / 255.0, axis=0)
            yield lr_y, hr_y

    def __len__(self):
        return len(self.all_images)


# Ref.: https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/6
class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


def get_data_loader(dirpath_train, dirpath_val, scaling_factor, patch_size, stride):
    """ Function to return train/val data loader

    :param dirpath_train (str): path to directory containing high resolution training images
    :param dirpath_val (str): path to directory containing high resolution validation images
    :param scaling_factor (int): Number by which to scale the lr image to hr image
    :param patch_size (int): size of sub-images extracted from original images
    :param stride (int): sub-images extraction stride
    :return (torch.utils.data.DataLoader): training and validation data loader
    """
    # As per paper, Sec. 3.2, sub-images are extracted only during the training phase
    dataset = SRTrainDataset(dirpath_images=dirpath_train,
                             scaling_factor=scaling_factor,
                             patch_size=patch_size,
                             stride=stride)
    train_dataset = ShuffleDataset(dataset, 1024)
    train_loader = DataLoader(train_dataset,
                              drop_last=True,
                              batch_size=16,
                              num_workers=4,
                              pin_memory=True)

    valid_dataset = SRValidDataset(dirpath_images=dirpath_val,
                                   scaling_factor=scaling_factor)
    val_loader = DataLoader(valid_dataset,
                            batch_size=1,
                            pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    # Test DataLoaders
    train_loader, val_loader = get_data_loader(dirpath_train="./dataset/train",
                                               dirpath_val="./dataset/val",
                                               scaling_factor=3,
                                               patch_size=17,
                                               stride=13)

    for idx, (lr_image, hr_image) in enumerate(train_loader):
        print(f"Training - lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        break

    for idx, (lr_image, hr_image) in enumerate(val_loader):
        print(f"Validation - lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        break

    for idx, (lr_image, hr_image) in enumerate(train_loader):
        print(f"lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        lr = lr_image[0].numpy().transpose(1, 2, 0)
        hr = hr_image[0].numpy().transpose(1, 2, 0)
        print(f"{lr.shape}, {hr.shape}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(lr)
        ax1.set_title("Low Res")
        ax2.imshow(hr)
        ax2.set_title("High Res")
        plt.show()
        break
