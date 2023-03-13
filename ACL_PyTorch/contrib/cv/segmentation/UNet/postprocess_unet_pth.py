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
# -*- coding: utf-8 -*-

import os
import time
import argparse

import numpy as np
from PIL import Image
import torch
import multiprocessing

from Pytorch_UNet.dice_loss import dice_coeff

gl_resDir = "new_result/bs1/"
gl_labelDir = "SegmentationClass/"


def getUnique(img):
    return np.unique(img)


def getIntersection(img, label, i):
    cnter = 0
    for h_img, h_label in zip(img, label):
        for w_img, w_label in zip(h_img, h_label):
            if w_img == i and w_label == i:
                cnter += 1
    return cnter


def getUnion(img, label, i):
    cnter = 0
    for h_img, h_label in zip(img, label):
        for w_img, w_label in zip(h_img, h_label):
            if w_img == i or w_label == i:
                cnter += 1
    return cnter


def getIoU(img, label):
    iou = 0.0
    cnter = 0
    uniqueVals = getUnique(img)
    for i in uniqueVals:
        if i == 0 or i > 21:
            continue
        intersection = getIntersection(img, label, i)
        union = getUnion(img, label, i)
        temp_iou = float(intersection) / union
        if temp_iou < 0.5:
            continue
        iou += temp_iou
        cnter += 1
    if cnter == 0:
        return 0
    else:
        return iou / cnter


def label_process(image, scale=1):

    image = Image.open(image)
    width, height = image.size
    width_scaled = int(width * scale)
    height_scaled = int(height * scale)
    image_scaled = image.resize((572, 572))
    image_array = np.array(image_scaled, dtype=np.uint8)

    return image_array


def postprocess(file):

    mask = torch.from_numpy(np.fromfile(os.path.join(gl_resDir, file), np.float32).reshape((572, 572)))
    mask = torch.sigmoid(mask)
    mask_array = (mask.numpy() > 0.5).astype(np.uint8)

    return mask_array


def eval_res(img_file, mask_file):

    image = torch.from_numpy(np.fromfile(os.path.join(gl_resDir, img_file), np.float32).reshape((572, 572)))
    image = torch.sigmoid(image)
    image = image > 0.5
    image = image.to(dtype=torch.float32)
    mask = Image.open(os.path.join(gl_labelDir, mask_file))
    mask = mask.resize((572, 572))
    mask = np.array(mask)
    mask = torch.from_numpy(mask)
    mask = mask.to(dtype=torch.float32)

    return dice_coeff(image, mask).item()


def get_iou(resLis_list, batch):
    sum_eval = 0.0
    for file in resLis_list[batch]:
        seval = eval_res(file, file.replace('.bin', '_mask.gif'))
        sum_eval += seval
        rVal = postprocess(file)
        lVal = label_process(os.path.join(gl_labelDir, file.replace('.bin', '_mask.gif')))
        iou = getIoU(rVal, lVal)
        if iou == 0:  # it's difficult
            continue
        print("    ---> {} IMAGE {} has IOU {}".format(batch, file, iou))
        lock.acquire()
        try:
            with open(gl_res_txt, 'a') as f:
                f.write('{}, '.format(iou))
        except:
            lock.release()
        lock.release()
    print("eval value is", sum_eval / len(resLis_list[batch]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='result/bs1')
    parser.add_argument('--label', default='./carvana/train_masks')
    parser.add_argument('--result', default='./result.txt')
    args = parser.parse_args()

    gl_resDir = args.output
    gl_labelDir = args.label
    gl_res_txt = args.result

    if gl_res_txt in os.listdir(os.getcwd()):
        os.remove(gl_res_txt)

    resLis = os.listdir(gl_resDir)
    resLis_list = [resLis[i:i + 300] for i in range(0, 5000, 300) if resLis[i:i + 300] != []]

    st = time.time()
    lock = multiprocessing.Lock()
    pool = multiprocessing.Pool(len(resLis_list))
    for batch in range(len(resLis_list)):
        pool.apply_async(get_iou, args=(resLis_list, batch))
    pool.close()
    pool.join()
    print('Multiple processes executed successfully')
    print('Time Used: {}'.format(time.time() - st))

    try:
        with open(gl_res_txt) as f:
            ret = list(map(float, f.read().replace(', ', ' ').strip().split(' ')))
        print('IOU Average ：{}'.format(sum(ret) / len(ret)))
    except:
        print('Failed to process data...')
