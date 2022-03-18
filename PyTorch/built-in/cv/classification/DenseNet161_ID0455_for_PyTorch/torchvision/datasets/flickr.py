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
#from collections import defaultdict
from PIL import Image
from six.moves import html_parser

import glob
import os
from .vision import VisionDataset


class Flickr8kParser(html_parser.HTMLParser):
    """Parser for extracting captions from the Flickr8k dataset web page."""

    def __init__(self, root):
        super(Flickr8kParser, self).__init__()

        self.root = root

        # Data structure to store captions
        self.annotations = {}

        # State variables
        self.in_table = False
        self.current_tag = None
        self.current_img = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

        if tag == 'table':
            self.in_table = True

    def handle_endtag(self, tag):
        self.current_tag = None

        if tag == 'table':
            self.in_table = False

    def handle_data(self, data):
        if self.in_table:
            if data == 'Image Not Found':
                self.current_img = None
            elif self.current_tag == 'a':
                img_id = data.split('/')[-2]
                img_id = os.path.join(self.root, img_id + '_*.jpg')
                img_id = glob.glob(img_id)[0]
                self.current_img = img_id
                self.annotations[img_id] = []
            elif self.current_tag == 'li' and self.current_img:
                img_id = self.current_img
                self.annotations[img_id].append(data.strip())


class Flickr8k(VisionDataset):
    """`Flickr8k Entities <http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, ann_file, transform=None, target_transform=None):
        super(Flickr8k, self).__init__(root, transform=transform,
                                       target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)

        # Read annotations and store in a dict
        parser = Flickr8kParser(self.root)
        with open(self.ann_file) as fh:
            parser.feed(fh.read())
        self.annotations = parser.annotations

        self.ids = list(sorted(self.annotations.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        img = Image.open(img_id).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


class Flickr30k(VisionDataset):
    """`Flickr30k Entities <http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, ann_file, transform=None, target_transform=None):
        super(Flickr30k, self).__init__(root, transform=transform,
                                        target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)

        # Read annotations and store in a dict
        self.annotations = defaultdict(list)
        with open(self.ann_file) as fh:
            for line in fh:
                img_id, caption = line.strip().split('\t')
                self.annotations[img_id[:-2]].append(caption)

        self.ids = list(sorted(self.annotations.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)
