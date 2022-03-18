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

import torch
import numpy as np
import PIL.Image as pil_image
import os
import argparse


parser = argparse.ArgumentParser(description='SRCNN post process script')
parser.add_argument('--res', default='', type=str, metavar='PATH',
                    help='om result path')
parser.add_argument('--png_src', default='', type=str, metavar='PATH',
                    help='png src path')
parser.add_argument('--bin_src', default='', type=str, metavar='PATH',
                    help='bin src path')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='result image save path')
args = parser.parse_args()

def postprocess(img_src_path, bin_src_path, src_path, save_path):
    total = 0;
    count = 0;
    # create dir
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for file in os.listdir(src_path):
        array = np.loadtxt(os.path.join(src_path, file),
                           np.float32).reshape(256, 256).transpose()

        for img_file in os.listdir(img_src_path):
            if img_file[0:-9] in file and "256" in img_file:
                tmp_image = pil_image.open(os.path.join(
                    img_src_path, img_file)).convert('RGB')
                tmp_image = np.array(tmp_image).astype(np.float32)
                ycbcr = convert_rgb_to_ycbcr(tmp_image)
                break
        # 统计精度
        for bin_file in os.listdir(bin_src_path):
            if bin_file[0:-4] in file:
                y = np.fromfile(os.path.join(
                    bin_src_path, bin_file), np.float32).reshape(256, 256).transpose()
                break

        torch_y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
        torch_array = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        psnr = calc_psnr(torch_y, torch_array)
        total += psnr
        print(img_file, end=' ')
        print('PSNR: {:.2f}'.format(psnr))
        print()
        # 输出
        array = torch.from_numpy(array).mul(255).numpy()
        output = np.array(
            [array, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output),
                         0.0, 255.0).astype(np.uint8)
        image = pil_image.fromarray(output)
        image.save(os.path.join(save_path+file)+".png")
        count += 1

    # calculate total psnr
    total /= count
    print('total PSNR: {:.2f}'.format(total))



def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 *
                   img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 *
                     img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 *
                     img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 *
                   img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 *
                     img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 *
                     img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + \
            408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :,
                                                          1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + \
            516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + \
            408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :,
                                                          :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + \
            516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


if __name__ == '__main__':
    postprocess(args.png_src, args.bin_src,
                args.res, args.save)
