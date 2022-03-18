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
from __future__ import division, print_function, absolute_import
import copy
import glob
import os.path as osp

from ..dataset import ImageDataset


class CUHKSYSU(ImageDataset):
    """CUHKSYSU.

    This dataset can only be used for model training.

    Reference:
        Xiao et al. End-to-end deep learning for person search.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html>`_
    
    Dataset statistics:
        - identities: 11,934
        - images: 34,574
    """
    _train_only = True
    dataset_dir = 'cuhksysu'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = osp.join(self.dataset_dir, 'cropped_images')

        # image name format: p11422_s16929_1.jpg
        train = self.process_dir(self.data_dir)
        query = [copy.deepcopy(train[0])]
        gallery = [copy.deepcopy(train[0])]

        super(CUHKSYSU, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dirname):
        img_paths = glob.glob(osp.join(dirname, '*.jpg'))
        # num_imgs = len(img_paths)

        # get all identities:
        pid_container = set()
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = img_name.split('_')[0]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # num_pids = len(pid_container)

        # extract data
        data = []
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = img_name.split('_')[0]
            label = pid2label[pid]
            data.append((img_path, label, 0)) # dummy camera id

        return data
