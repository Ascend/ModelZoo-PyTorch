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

from torchreid.utils import read_json

from ..dataset import VideoDataset


class PRID2011(VideoDataset):
    """PRID2011.

    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and
        Discriminative Classification. SCIA 2011.

    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_
    
    Dataset statistics:
        - identities: 200.
        - tracklets: 400.
        - cameras: 2.
    """
    dataset_dir = 'prid2011'
    dataset_url = None

    def __init__(self, root='', split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.split_path = osp.join(self.dataset_dir, 'splits_prid2011.json')
        self.cam_a_dir = osp.join(
            self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_a'
        )
        self.cam_b_dir = osp.join(
            self.dataset_dir, 'prid_2011', 'multi_shot', 'cam_b'
        )

        required_files = [self.dataset_dir, self.cam_a_dir, self.cam_b_dir]
        self.check_before_run(required_files)

        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                .format(split_id,
                        len(splits) - 1)
            )
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']

        train = self.process_dir(train_dirs, cam1=True, cam2=True)
        query = self.process_dir(test_dirs, cam1=True, cam2=False)
        gallery = self.process_dir(test_dirs, cam1=False, cam2=True)

        super(PRID2011, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))

            if cam2:
                person_dir = osp.join(self.cam_b_dir, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))

        return tracklets
