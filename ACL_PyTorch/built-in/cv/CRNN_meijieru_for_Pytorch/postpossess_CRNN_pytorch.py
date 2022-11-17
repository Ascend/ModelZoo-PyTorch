# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import sys
import numpy as np
import torch
from torch.autograd import Variable
import utils

alphabets = '0123456789abcdefghijklmnopqrstuvwxyz'

def postpossess(bin_path):
    converter = utils.strLabelConverter(alphabets)
    preds = np.fromfile(bin_path, dtype=np.float32)
    preds = preds.reshape(26,-1,37)
    preds = torch.from_numpy(preds)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

if __name__ == '__main__':
    bin_path = sys.argv[1]
    postpossess(bin_path)
