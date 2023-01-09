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

import os
import sys
import numpy as np
import torch
import cv2
from pypse import pse as pypse



img_path = sys.argv[1]
bin_path = sys.argv[2]
txt_path = sys.argv[3]

if not os.path.exists(txt_path):
    os.makedirs(txt_path)

kernel_num=7
min_kernel_area=5.0
scale=1
min_score = 0.9
min_area = 600


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']

    for parent, _, filenames in os.walk(img_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    return files


im_fn_list = get_images()
for im_fn in im_fn_list:
    im = cv2.imread(im_fn)
    idx = os.path.basename(im_fn).split('/')[-1].split('.')[0].split('_')[1]
    seg_maps = np.fromfile(bin_path+"/img_{}_0.bin".format(idx), "float32")
    seg_maps = np.reshape(seg_maps, (1, 7, 704, 1216))
    seg_maps = torch.from_numpy(seg_maps)

    
    score = torch.sigmoid(seg_maps[:, 0, :, :])
    outputs = (torch.sign(seg_maps - 1.0) + 1) / 2

    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:kernel_num, :, :] * text

    score = score.data.numpy()[0].astype(np.float32)
    text = text.data.numpy()[0].astype(np.uint8)
    kernels = kernels.numpy()[0].astype(np.uint8)

    # python version pse
    pred = pypse(kernels, min_kernel_area / (scale * scale))

    img_scale = (im.shape[1] * 1.0 / pred.shape[1], im.shape[0] * 1.0 / pred.shape[0])
    label = pred
    label_num = np.max(label) + 1
    bboxes = []

    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < min_area:
            continue

        score_i = np.mean(score[label == i])
        if score_i < min_score:
            continue

        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect) * img_scale
        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1))
    
    # save txt
    res_file = os.path.join(txt_path,'{}.txt'.format(os.path.splitext(os.path.basename(im_fn))[0]))
    with open(res_file, 'w') as f:
        for b_idx, bbox in enumerate(bboxes):
            values = [int(v) for v in bbox]
            line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
            f.write(line)


   


