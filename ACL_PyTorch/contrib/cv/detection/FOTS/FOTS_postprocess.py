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

import argparse
import os

import numpy as np
import torch

from modules.parse_polys import parse_polys
import re
import tqdm
import os
import sys
import struct


# bin文件格式转为numpy
def bin2np(binName, binShape):
    size = os.path.getsize(binName)     # size 是字节大小
    binfile = open(binName, 'rb')
    Len = int(size / 4)                 # 4个字节=float32 类型
    res = []
    for i in range(Len):
        data = binfile.read(4)  # 将4个字节取出作为 float
        num = struct.unpack('f', data)
        res.append(num[0])

    binfile.close()

    dim_res = np.array(res).reshape(binShape)

    return dim_res


# bin 文件转回 tensor
def postprocess(bin_folder, output_folder):

    preNum = 1
    while (preNum < 501):

        scale_x = 2240 / 1280
        scale_y = 1248 / 720

        preName = "img_" + str(preNum)
        confBin = bin_folder + preName + "_0.bin"
        disBin = bin_folder + preName + "_1.bin"
        angleBin = bin_folder + preName + "_2.bin"
        preNum += 1

        confidence = torch.tensor(bin2np(confBin, (1, 1, 312, 560)))
        distances = torch.tensor(bin2np(disBin, (1, 4, 312, 560)))
        angle = torch.tensor(bin2np(angleBin, (1, 1, 312, 560)))

        confidence = torch.sigmoid(confidence).squeeze().data.cpu().numpy()
        distances = distances.squeeze().data.cpu().numpy()
        angle = angle.squeeze().data.cpu().numpy()

        polys = parse_polys(confidence, distances, angle, 0.95, 0.3)
        with open('{}'.format(os.path.join(output_folder, 'res_{}.txt'.format(preName))), 'w') as f:
            for id in range(polys.shape[0]):
                f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                    int(polys[id, 0] / scale_x), int(polys[id, 1] /
                                                     scale_y), int(polys[id, 2] / scale_x),
                    int(polys[id, 3] / scale_y),
                    int(polys[id, 4] / scale_x), int(polys[id, 5] /
                                                     scale_y), int(polys[id, 6] / scale_x),
                    int(polys[id, 7] / scale_y)
                ))


if __name__ == '__main__':

    output_folder = sys.argv[1]
    bin_folder = sys.argv[2]

    postprocess(bin_folder, output_folder)
