# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
from models import Generator
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import argparse

def main(args):
    os.makedirs(args.online_path, exist_ok=True)
    os.makedirs(args.offline_path, exist_ok=True)
    generator = Generator()
    pre = torch.load(args.pth_path,map_location='cpu')

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in pre.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    # load params
    generator.load_state_dict(new_state_dict)
    Tensor = torch.FloatTensor
    for i in range(args.iters):
        z = Variable(Tensor(np.random.normal(0, 1, (args.batch_size,100))))

        if args.batch_size != 1:
            gen = generator(z)
            save_image(gen, args.online_path+"/%d.jpg" % i,normalize=True)

        z = z.numpy()
        z.tofile(args.offline_path+"/%d.bin"% i)

    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--online_path', type=str, required=True)
    parser.add_argument('--offline_path', type=str, required=True)
    parser.add_argument('--pth_path', type=str, required=True)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    main(args)