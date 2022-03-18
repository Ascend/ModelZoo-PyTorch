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
import numpy as np
import PIL.Image as pil_image
import os
import argparse
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr


parser = argparse.ArgumentParser(description='SRCNN post process script')

parser.add_argument('--hr', default='', type=str, metavar='PATH',
                    help='hr path')
parser.add_argument('--binres', default='', type=str, metavar='PATH',
                    help='bin result path')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='result image save path')
args = parser.parse_args()


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def postprocess(hr, binres, save_path):
    # create dir
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    avg_psnr = 0
    number = 0
    files = os.listdir(hr)
    files.sort()
    for file in files:
        hr_image = pil_image.open(os.path.join(hr, file)).convert('RGB')

        hr_image = np.array(hr_image).astype(np.float32)
        # print(hr_image)
        for bin_file in os.listdir(binres):
            if file[0:-4] in bin_file and bin_file[-6:] == '_1.bin':
                y = np.fromfile(os.path.join(
                    binres, bin_file), np.float32).reshape(3, 2048, 2048)
                y = (np.clip(y, 0, 1) * 255).astype(np.uint8)

                break

        torch_hr = hr_image
        torch_y = y.transpose([2, 1, 0])[
            :torch_hr.shape[0], :torch_hr.shape[1], :]

        psnr_val = psnr(torch_y, torch_hr)
        print(file, 'PSNR: {:.2f}'.format(psnr_val))
        avg_psnr += psnr_val
        number += 1
        imwrite(os.path.join(save_path+file), np.array(torch_y))
    avg_psnr = avg_psnr/number
    print('avg_psnr:', avg_psnr)


if __name__ == '__main__':
    postprocess(args.hr, args.binres, args.save)
