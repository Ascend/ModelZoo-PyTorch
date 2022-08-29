# Copyright 2022 Huawei Technologies Co., Ltd
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
from __future__ import print_function
import argparse
import torch
import os
from glob import glob
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np
'''
Original : https://github.com/pytorch/examples/tree/master/super_resolution

'''


class inference:
    def __init__(self):
    # Training settings
        parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
        parser.add_argument('--input_image', type=str, required=True, help='input image to use')
        parser.add_argument('--model', type=str, required=True, help='model file to use')
        parser.add_argument('--output_filename', type=str, help='where to save the output image')
        parser.add_argument('--cuda', action='store_true', help='use cuda')
        opt = parser.parse_args()

        def get_most_recent_checkpoint(checkpoint_dir):
            checkpoint_paths = [path for path in glob("{}/model_epoch_*.pth".format(checkpoint_dir))]
            idxes = [int(os.path.basename(path).split('_')[2].split('.')[0]) for path in checkpoint_paths]

            max_idx = max(idxes)
            latest_checkpoint = os.path.join(checkpoint_dir, "model_epoch_{}.pth".format(max_idx))
            print(" [*] Found latest checkpoint: {}".format(latest_checkpoint))
            return latest_checkpoint, max_idx

        print("=============> Load model")
        print(opt)
        img = Image.open(opt.input_image).convert('YCbCr')
        y, cb, cr = img.split()

        last_ckpt , _ = get_most_recent_checkpoint(opt.model)
        model = torch.load(last_ckpt)
        img_to_tensor = ToTensor()
        input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
        print("=============> Load model Complete")
        print("=============> Input load ")

        # if opt.cuda:
        #     model = model.cuda()
        #     input = input.cuda()
        print("=============> Input load Complete")
        print("=============> Inference start")
        out = model(input)
        out = out.cpu()
        print("=============> Inference Complete")
        print("=============> Start Merge")

        out_img_y = out[0].detach().numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        print("=============> Merge  Complete")

        out_img.save(opt.output_filename)
        print('output image saved to ', opt.output_filename)


if __name__ == '__main__':
    inference()
