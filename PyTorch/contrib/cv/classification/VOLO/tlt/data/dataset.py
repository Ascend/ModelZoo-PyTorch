"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import re
import torch
import tarfile
import logging
from PIL import Image
_logger = logging.getLogger('token_label_dataset')

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


def load_class_map(filename, root=''):
    class_map_path = filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, filename)
        assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % filename
    class_map_ext = os.path.splitext(filename)[-1].lower()
    if class_map_ext == '.txt':
        with open(class_map_path) as f:
            class_to_idx = {v.strip(): k for k, v in enumerate(f)}
    else:
        assert False, 'Unsupported class map extension'
    return class_to_idx


class DatasetTokenLabel(data.Dataset):

    def __init__(
            self,
            root,
            label_root,
            load_bytes=False,
            transform=None,
            class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        images, class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.label_root = label_root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        score_path = os.path.join(
            self.label_root,
            '/'.join(path.split('/')[-2:]).split('.')[0] + '.pt')

        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        score_maps = torch.load(score_path).float()
        if self.transform is not None:
            img, score_maps = self.transform(img, score_maps)
        # append ground truth after coords
        score_maps[-1,0,0,5]=target
        return img, score_maps

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.samples]


def create_token_label_dataset(dataset_type, root, label_root):
    train_dir = os.path.join(root, 'train')
    if not os.path.exists(train_dir):
        _logger.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    if not os.path.exists(label_root):
        _logger.error('Label folder does not exist at: {}'.format(label_root))
        exit(1)
    return DatasetTokenLabel(train_dir, label_root)
