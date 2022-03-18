# coding: utf-8
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
""" MSCOCO Pytorch Dataset Class"""

from torch.utils.data import Dataset
from py_cocodata_server.py_data_iterator import RawDataIterator
import numpy as np
from config.config import GetConfig, COCOSourceConfig
from time import time
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import torch


class MyDataset(Dataset):
    def __init__(self, global_config, config, shuffle=False, augment=True):
        """
        Initialize a DataIterator
        :param global_config: the configuration used in our project
        :param config:  the original COCO configuration
        :param shuffle:
        :param augment:
        """
        self.global_config = global_config
        self.config = config
        self.shuffle = shuffle
        self.augment = augment
        self.raw_data_iterator = RawDataIterator(self.global_config, self.config, shuffle=self.shuffle,
                                                 augment=self.augment)

    def __getitem__(self, index):
        # return entries: image, mask_miss, unmasked labels, offsets, mask_offset
        # Notice锛 numpy.random seed will fork the same value in multi-process, while python random will fork differently
        return self.raw_data_iterator.gen(index)

    def __len__(self):
        return self.raw_data_iterator.num_keys()


if __name__ == '__main__':  # for debug

    def test_augmentation_speed(train_client, show_image=True):
        start = time()
        batch = 0
        for index in range(train_client.__len__()):
            batch += 1

            image, mask_miss, labels = [v.numpy() for v in    # , offsets, mask_offset
                                            train_client.__getitem__(index)]

            # show the generated ground truth
            if show_image:
                show_labels = cv2.resize(labels.transpose((1, 2, 0)), image.shape[:2], interpolation=cv2.INTER_CUBIC)
                # offsets = cv2.resize(offsets.transpose((1, 2, 0)), image.shape[:2], interpolation=cv2.INTER_NEAREST)
                mask_miss = np.repeat(mask_miss.transpose((1, 2, 0)), 3, axis=2)
                # mask_miss = cv2.resize(mask_miss, image.shape[:2], interpolation=cv2.INTER_NEAREST)
                image = cv2.resize(image, mask_miss.shape[:2], interpolation=cv2.INTER_NEAREST)
                plt.imshow(image[:, :, [2, 1, 0]])   # Opencv image format: BGR
                plt.imshow(labels.transpose((1, 2, 0))[:,:,20], alpha=0.5)  # mask_all
                # plt.imshow(show_labels[:, :, 3], alpha=0.5)  # mask_all
                plt.show()
                t=2
        print("produce %d samples per second: " % (batch / (time() - start)))

    config = GetConfig("Canonical")
    soureconfig = COCOSourceConfig("../data/dataset/coco/link2coco2017/coco_val_dataset384.h5")

    val_client = MyDataset(config, soureconfig, shuffle=True, augment=True)  # shuffle in data loader
    # test the data generator
    test_augmentation_speed(val_client, True)
