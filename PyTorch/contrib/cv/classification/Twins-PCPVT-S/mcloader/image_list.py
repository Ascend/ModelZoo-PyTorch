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
from PIL import Image

from .mcloader import McLoader


class ImageList(object):

    def __init__(self, root, list_file, memcached=False, mclient_path=None):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.has_labels = len(lines[0].split()) == 2
        if self.has_labels:
            self.fns, self.labels = zip(*[l.strip().split() for l in lines])
            self.labels = [int(l) for l in self.labels]
        else:
            self.fns = [l.strip() for l in lines]
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path)
            self.initialized = True

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img = self.mc_loader(self.fns[idx])
        else:
            img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        if self.has_labels:
            target = self.labels[idx]
            return img, target
        else:
            return img
