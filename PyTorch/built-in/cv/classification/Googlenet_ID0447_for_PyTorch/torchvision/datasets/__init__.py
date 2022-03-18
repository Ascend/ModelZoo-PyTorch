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
#from .lsun import LSUN, LSUNClass
from .folder import ImageFolder, DatasetFolder
from .coco import CocoCaptions, CocoDetection
from .cifar import CIFAR10, CIFAR100
from .stl10 import STL10
from .mnist import MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST
from .svhn import SVHN
from .phototour import PhotoTour
from .fakedata import FakeData
from .semeion import SEMEION
from .omniglot import Omniglot
from .sbu import SBU
from .flickr import Flickr8k, Flickr30k
from .voc import VOCSegmentation, VOCDetection
from .cityscapes import Cityscapes
from .imagenet import ImageNet
from .caltech import Caltech101, Caltech256
from .celeba import CelebA
from .sbd import SBDataset
from .vision import VisionDataset
from .usps import USPS
from .kinetics import Kinetics400
from .hmdb51 import HMDB51
from .ucf101 import UCF101

__all__ = ('LSUN', 'LSUNClass',
           'ImageFolder', 'DatasetFolder', 'FakeData',
           'CocoCaptions', 'CocoDetection',
           'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
           'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
           'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
           'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
           'Caltech101', 'Caltech256', 'CelebA', 'SBDataset', 'VisionDataset',
           'USPS', 'Kinetics400', 'HMDB51', 'UCF101')
