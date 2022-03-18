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


class SenseReID(ImageDataset):
    """SenseReID.

    This dataset is used for test purpose only.

    Reference:
        Zhao et al. Spindle Net: Person Re-identification with Human Body
        Region Guided Feature Decomposition and Fusion. CVPR 2017.

    URL: `<https://drive.google.com/file/d/0B56OfSrVI8hubVJLTzkwV2VaOWM/view>`_

    Dataset statistics:
        - query: 522 ids, 1040 images.
        - gallery: 1717 ids, 3388 images.
    """
    dataset_dir = 'sensereid'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.query_dir = osp.join(self.dataset_dir, 'SenseReID', 'test_probe')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'SenseReID', 'test_gallery'
        )

        required_files = [self.dataset_dir, self.query_dir, self.gallery_dir]
        self.check_before_run(required_files)

        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        # relabel
        g_pids = set()
        for _, pid, _ in gallery:
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        query = [
            (img_path, pid2label[pid], camid) for img_path, pid, camid in query
        ]
        gallery = [
            (img_path, pid2label[pid], camid)
            for img_path, pid, camid in gallery
        ]
        train = copy.deepcopy(query) + copy.deepcopy(gallery) # dummy variable

        super(SenseReID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        data = []

        for img_path in img_paths:
            img_name = osp.splitext(osp.basename(img_path))[0]
            pid, camid = img_name.split('_')
            pid, camid = int(pid), int(camid)
            data.append((img_path, pid, camid))

        return data
