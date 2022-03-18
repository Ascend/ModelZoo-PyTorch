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
import glob
import os.path as osp
import warnings

from torchreid.utils import read_json, write_json

from ..dataset import VideoDataset


class DukeMTMCVidReID(VideoDataset):
    """DukeMTMCVidReID.

    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target,
          Multi-Camera Tracking. ECCVW 2016.
        - Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
          Re-Identification by Stepwise Learning. CVPR 2018.

    URL: `<https://github.com/Yu-Wu/DukeMTMC-VideoReID>`_
    
    Dataset statistics:
        - identities: 702 (train) + 702 (test).
        - tracklets: 2196 (train) + 2636 (test).
    """
    dataset_dir = 'dukemtmc-vidreid'
    dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip'

    def __init__(self, root='', min_seq_len=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.train_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/train')
        self.query_dir = osp.join(self.dataset_dir, 'DukeMTMC-VideoReID/query')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'DukeMTMC-VideoReID/gallery'
        )
        self.split_train_json_path = osp.join(
            self.dataset_dir, 'split_train.json'
        )
        self.split_query_json_path = osp.join(
            self.dataset_dir, 'split_query.json'
        )
        self.split_gallery_json_path = osp.join(
            self.dataset_dir, 'split_gallery.json'
        )
        self.min_seq_len = min_seq_len

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(
            self.train_dir, self.split_train_json_path, relabel=True
        )
        query = self.process_dir(
            self.query_dir, self.split_query_json_path, relabel=False
        )
        gallery = self.process_dir(
            self.gallery_dir, self.split_gallery_json_path, relabel=False
        )

        super(DukeMTMCVidReID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, json_path, relabel):
        if osp.exists(json_path):
            split = read_json(json_path)
            return split['tracklets']

        print('=> Generating split json file (** this might take a while **)')
        pdirs = glob.glob(osp.join(dir_path, '*')) # avoid .DS_Store
        print(
            'Processing "{}" with {} person identities'.format(
                dir_path, len(pdirs)
            )
        )

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        tracklets = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel:
                pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx + 1).zfill(4)
                    res = glob.glob(
                        osp.join(tdir, '*' + img_idx_name + '*.jpg')
                    )
                    if len(res) == 0:
                        warnings.warn(
                            'Index name {} in {} is missing, skip'.format(
                                img_idx_name, tdir
                            )
                        )
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))

        print('Saving split to {}'.format(json_path))
        split_dict = {'tracklets': tracklets}
        write_json(split_dict, json_path)

        return tracklets
