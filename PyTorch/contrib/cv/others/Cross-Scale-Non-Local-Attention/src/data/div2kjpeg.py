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
from data import srdata
from data import div2k

class DIV2KJPEG(div2k.DIV2K):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.q_factor = int(name.replace('DIV2K-Q', ''))
        super(DIV2KJPEG, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'DIV2K')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(
            self.apath, 'DIV2K_Q{}'.format(self.q_factor)
        )
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.jpg')

