# MIT License
#
# Copyright (c) 2018 kuangliu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
#
# MIT License
#
# Copyright (c) 2017 Max deGroot, Ellis Brown
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
#
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from xml.etree import ElementTree
import os
import glob
from pathlib import Path
import numpy as np
import random
import itertools
import torch.nn.functional as F
try:
    import ujson as json
except ImportError:
    import json
import gc
import time
import bz2
import pickle
from math import sqrt, ceil, cos, sin, pi
from mlperf_logging.mllog import constants
from mlperf_logger import log_event
from box_coder import Encoder
# from SSD import _C as C
from box_coder import dboxes300_coco,build_ssd300_coder

# This function is from https://github.com/kuangliu/pytorch-ssd
def calc_iou_tensor(box1, box2):
    """ Calculation of IoU based on two boxes tensor,
        Reference to https://github.com/kuangliu/pytorch-ssd
        input:
            box1 (N, 4)
            box2 (M, 4)
        output:
            IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(be1[:,:,:2], be2[:,:,:2])
    #mask1 = (be1[:,:, 0] < be2[:,:, 0]) ^ (be1[:,:, 1] < be2[:,:, 1])
    #mask1 = ~mask1
    rb = torch.min(be1[:,:,2:], be2[:,:,2:])
    #mask2 = (be1[:,:, 2] < be2[:,:, 2]) ^ (be1[:,:, 3] < be2[:,:, 3])
    #mask2 = ~mask2

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:,:,0]*delta[:,:,1]
    #*mask1.float()*mask2.float()

    delta1 = be1[:,:,2:] - be1[:,:,:2]
    area1 = delta1[:,:,0]*delta1[:,:,1]
    delta2 = be2[:,:,2:] - be2[:,:,:2]
    area2 = delta2[:,:,0]*delta2[:,:,1]

    iou = intersect/(area1 + area2 - intersect)
    return iou

# This class is from https://github.com/chauhan-utk/ssd.DomainAdaptation
class SSDCropping(object):
    """ Cropping for SSD, according to original paper
        Choose between following 3 conditions:
        1. Preserve the original image
        2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
        3. Random crop
        Reference to https://github.com/chauhan-utk/ssd.DomainAdaptation
    """
    def __init__(self):

        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )
        # Implementation uses 1 iteration to find a possible candidate, this
        # was shown to produce the same mAP as using more iterations.
        self.num_cropping_iterations = 1
        log_event(key=constants.MAX_SAMPLES,
                  value=self.num_cropping_iterations)

    def __call__(self, img, img_size, bboxes, labels):

        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)

            if mode is None:
                return img, img_size, bboxes, labels

            htot, wtot = img_size

            min_iou, max_iou = mode
            min_iou = float("-inf") if min_iou is None else min_iou
            max_iou = float("+inf") if max_iou is None else max_iou

            # Implementation use 50 iteration to find possible candidate
            for _ in range(self.num_cropping_iterations):
                # suze of each sampled path in [0.1, 1] 0.3*0.3 approx. 0.1
                w = random.uniform(0.3 , 1.0)         
                h = random.uniform(0.3 , 1.0)        

                if w/h < 0.5 or w/h > 2:
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)       
                top = random.uniform(0, 1.0 - h)           

                right = left + w
                bottom = top + h
                tensor = torch.tensor([[left, top, right, bottom]]).cpu()
                ious = calc_iou_tensor(bboxes, tensor)
                # tailor all the bboxes and return
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5*(bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5*(bboxes[:, 1] + bboxes[:, 3])

                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes, continue searching again
                if not masks.any():
                    continue

                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom

                bboxes = bboxes[masks, :]
                labels = labels[masks]

                left_idx = int(left*wtot)
                top_idx =  int(top*htot)
                right_idx = int(right*wtot)
                bottom_idx = int(bottom*htot)
                img = img.crop((left_idx, top_idx, right_idx, bottom_idx))

                bboxes[:, 0] = (bboxes[:, 0] - left)/w
                bboxes[:, 1] = (bboxes[:, 1] - top)/h
                bboxes[:, 2] = (bboxes[:, 2] - left)/w
                bboxes[:, 3] = (bboxes[:, 3] - top)/h

                htot = bottom_idx - top_idx
                wtot = right_idx - left_idx
                return img, (htot, wtot), bboxes, labels

# Don't need to cast to float, already there (from FusedColorJitter)
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = torch.Tensor(np.array(img)).cpu()                      #########################
        # Transform from HWC to CHW
        img = img.permute(2, 0 ,1).div(255)
        return img

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, bboxes):
        if random.random() < self.p:
            bboxes[:, 0], bboxes[:, 2] = 1.0 - bboxes[:, 2], 1.0 - bboxes[:, 0]
            return image.transpose(Image.FLIP_LEFT_RIGHT), bboxes
        return image, bboxes

# Do data augumentation
class SSDTransformer(object):
    """ SSD Data Augumentation, according to original paper
        Composed by several steps:
        Cropping
        Resize
        Flipping
        Jittering
    """
    def __init__(self, size = (300, 300), val=False):

        # define vgg16 mean
        self.size = size
        self.val = val

        self.crop = SSDCropping()
        self.img_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ColorJitter(brightness=0.125, contrast=0.5,
                saturation=0.5, hue=0.05
            ),
            transforms.ToTensor(),
            #FusedColorJitter(),
        ])
        self.hflip = RandomHorizontalFlip()

        # All Pytorch Tensor will be normalized
        # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683

        normalization_mean = [0.485, 0.456, 0.406]
        normalization_std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=normalization_mean,
                                              std=normalization_std)

        self.trans_val = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            self.normalize,])

    def __call__(self, img, img_size, bbox=None, label=None, max_num=200):
        #img = torch.tensor(img)
        if self.val:
            bbox_out = torch.zeros(max_num, 4)
            label_out =  torch.zeros(max_num, dtype=torch.long)
            bbox_out[:bbox.size(0), :] = bbox
            label_out[:label.size(0)] = label
            return self.trans_val(img), img_size, bbox_out, label_out

        # random crop
        img, img_size, bbox, label = self.crop(img, img_size, bbox, label)

        # random horiz. flip
        img, bbox = self.hflip(img, bbox)

        # [Resize, ColorJitter, ToTensor]
        img = self.img_trans(img).contiguous()

        img = self.normalize(img)

        return img, img_size, bbox, label
        

# Implement a datareader for COCO dataset
class COCODetection(data.Dataset):
    def __init__(self, img_folder, annotate_file, transform=None, data=None):
        self.img_folder = img_folder
        self.annotate_file = annotate_file

        if data:
            self.data = data
        else:
            # Start processing annotation
            with open(annotate_file) as fin:
                # loading huge json files tends to cause the gc (cycle collector) to
                # waste a lot of time so:
                gc_old = gc.isenabled()
                gc.disable()

                self.data = json.load(fin)

                if gc_old: gc.enable()

        self.images = {}

        self.label_map = {}
        self.label_info = {}
        start_time = time.time()
        # 0 stand for the background
        cnt = 0
        self.label_info[cnt] = "background"
        for cat in self.data["categories"]:
            cnt += 1
            self.label_map[cat["id"]] = cnt
            self.label_info[cnt] = cat["name"]

        # build inference for images
        for img in self.data["images"]:
            img_id = img["id"]
            img_name = img["file_name"]
            img_size = (img["height"],img["width"])
            if img_id in self.images: raise Exception("dulpicated image record")
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data["annotations"]:
            img_id = bboxes["image_id"]
            category_id = bboxes["category_id"]
            bbox = bboxes["bbox"]
            bbox_label = self.label_map[bboxes["category_id"]]
            self.images[img_id][2].append((bbox, bbox_label))

        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                self.images.pop(k)

        self.img_keys = list(self.images.keys())
        self.transform = transform

        self.encoder = Encoder(dboxes300_coco())


    @property
    def labelnum(self):
        return len(self.label_info)

    @staticmethod
    def load(pklfile):
        with bz2.open(pklfile, "rb") as fin:
            ret = pickle.load(fin)
        return ret

    def save(self, pklfile):
        with bz2.open(pklfile, "wb") as fout:
            pickle.dump(self, fout)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        fn = img_data[0]
        img_path = os.path.join(self.img_folder, fn)
        s = time.time()
        img = Image.open(img_path).convert("RGB")
        e = time.time()
        decode_time = e - s

        htot, wtot = img_data[1]
        bbox_sizes = []
        bbox_labels = []

        for (l,t,w,h), bbox_label in img_data[2]:
            r = l + w
            b = t + h
            #l, t, r, b = xc - 0.5*w, yc - 0.5*h, xc + 0.5*w, yc + 0.5*h
            bbox_size = (l/wtot, t/htot, r/wtot, b/htot)
            # filter out zero-size bboxes
            if l == r or t == b:
                continue
            bbox_sizes.append(bbox_size)
            bbox_labels.append(bbox_label)

        bbox_sizes = torch.tensor(bbox_sizes)
        bbox_labels =  torch.tensor(bbox_labels)

        s = time.time()
        if self.transform != None:
            img, (htot, wtot), bbox_sizes, bbox_labels = \
                self.transform(img, (htot, wtot), bbox_sizes, bbox_labels)
        else:
            img = transforms.ToTensor()(img)


        bbox_sizes, bbox_labels = self.encoder.encode(bbox_sizes, bbox_labels)


        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels
