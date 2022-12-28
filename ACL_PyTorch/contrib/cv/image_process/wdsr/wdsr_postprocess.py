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
import argparse
import os
import torch
import numpy as np
from PIL import Image

def psnr(sr, hr, shave=4):
  sr = sr.to(hr.dtype)
  sr = (sr * 255).round().clamp(0, 255) / 255
  diff = sr - hr
  if shave:
    diff = diff[..., shave:-shave, shave:-shave]
  mse = diff.pow(2).mean([-3, -2, -1])
  psnr = -10 * mse.log10()
  return psnr.mean()


def unPackPad(lr_image, width, height):
    lr_image = lr_image.reshape((3, 2040, 2040))
    lr_height = lr_image.shape[2]
    lr_width = lr_image.shape[1]

    pad_w = int((lr_width - width) / 2)
    pad_h = int((lr_height - height) / 2)

    lr_image_unpad = lr_image[:, pad_h:2040 - pad_h, pad_w:2040 - pad_w]

    return lr_image_unpad


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bin_data_path',
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        '--dataset_path',
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        '--result',
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        '--scale',
        default=None,
        type=int,
        required=True,
    )
    arg, _ = parser.parse_known_args()

    binData = os.listdir(arg.bin_data_path)
    datasetData = os.listdir(arg.dataset_path)
    datasetData.sort()
    shave = 0
    if arg.scale != 1:
        shave = arg.scale + 6

    sum = 0
    avg = 0.0
    count = 0
    with open(arg.result,'w') as resultFile:
        for lrFile in os.listdir(arg.bin_data_path):
            hrFile = lrFile[:4] + '.png'
            srcData = np.fromfile(os.path.join(arg.bin_data_path, lrFile), dtype="float32")
            srcData = srcData.reshape((3, 2040, 2040))
            hr = Image.open(os.path.join(arg.dataset_path, hrFile))
            srcData = unPackPad(srcData, hr.width, hr.height)
            hr = np.asarray(hr)
            hr = hr.transpose((2, 0, 1))
            hr = hr.astype(np.float32) / 255.
            srcDataTensor = torch.from_numpy(srcData)
            binDataTensor = torch.from_numpy(hr)
            result = psnr(srcDataTensor, binDataTensor, shave)
            sum += result.item()
            count += 1
            avg = sum / count
            content = hrFile+' PSNR: {:.2f}'.format(result.item())
            print(content)
            resultFile.write(content)
            resultFile.write('\n')

        print('avg_psnr:', avg)
        resultFile.write('avg_psnr:'+str(avg))
