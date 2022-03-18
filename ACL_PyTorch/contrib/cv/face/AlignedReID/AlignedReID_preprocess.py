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

from __future__ import print_function
import os
import sys
sys.path.insert(0, './AlignedReID-Re-Production-Pytorch')
import numpy as np
from PIL import Image
import argparse

from aligned_reid.dataset import create_dataset
from aligned_reid.utils.utils import str2bool


class Config(object):

    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--set_seed', type=str2bool, default=True)
        parser.add_argument('--dataset', type=str, default='market1501',
                            choices=['market1501', 'cuhk03', 'duke', 'combined'])
        parser.add_argument('--trainset_part', type=str, default='trainval',
                            choices=['trainval', 'train'])

        # Only for training set.
        parser.add_argument('--resize_h_w', type=eval, default=(256, 128))

        args = parser.parse_known_args()[0]

        if args.set_seed:
            self.seed = 1
        else:
            self.seed = None

        ###########
        # Dataset #
        ###########

        # If you want to exactly reproduce the result in training, you have to set
        # num of threads to 1.
        if self.seed is not None:
            self.prefetch_threads = 1
        else:
            self.prefetch_threads = 2

        self.dataset = args.dataset

        # Image Processing
        self.resize_h_w = args.resize_h_w

        self.scale_im = True
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]

        self.test_batch_size = 32
        self.test_final_batch = True
        self.test_mirror_type = ['random', 'always', None][2]
        self.test_shuffle = False

        dataset_kwargs = dict(
            name=self.dataset,
            resize_h_w=self.resize_h_w,
            scale=self.scale_im,
            im_mean=self.im_mean,
            im_std=self.im_std,
            batch_dims='NCHW',
            num_prefetch_threads=self.prefetch_threads)
        prng = np.random
        self.test_set_kwargs = dict(
            part='test',
            batch_size=self.test_batch_size,
            final_batch=self.test_final_batch,
            shuffle=self.test_shuffle,
            mirror_type=self.test_mirror_type,
            prng=prng)
        self.test_set_kwargs.update(dataset_kwargs)


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def preprocess(file_path, bin_path):
    cfg = Config()
    test_sets = create_dataset(**cfg.test_set_kwargs)

    in_files = os.listdir(file_path)
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    i = 0

    resize_size = (256, 128)
    mean = [0.486, 0.459, 0.408]
    std = [0.229, 0.224, 0.225]

    for file in in_files:
        if file in test_sets.im_names:
            i = i + 1
            print(file, "===", i)

            img = Image.open(os.path.join(file_path, file)).convert('RGB')

            img = resize(img, resize_size)
            img = np.array(img, dtype=np.float32)
            img = img / 255.
            # Mean variance
            img[..., 0] -= mean[0]
            img[..., 1] -= mean[1]
            img[..., 2] -= mean[2]
            img[..., 0] /= std[0]
            img[..., 1] /= std[1]
            img[..., 2] /= std[2]

            img = img[:, ::-1, :]

            img = img.transpose(2, 0, 1)  # HWC -> CHW

            img.tofile(os.path.join(bin_path, file.split('.')[0] + '.bin'))


if __name__ == "__main__":
    file_path = os.path.abspath(sys.argv[1])
    bin_path = os.path.abspath(sys.argv[2])
    preprocess(file_path, bin_path)
