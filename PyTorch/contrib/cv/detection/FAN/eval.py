# Copyright 2022 Huawei Technologies Co., Ltd
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
import face_alignment
import numpy as np
from sklearn.metrics import roc_auc_score, explained_variance_score
from tqdm import tqdm
from dataset import ImgDataset
import matplotlib.pyplot as plt
import torch
from utils import AverageMeter
import time
import argparse
from skimage import io
if torch.__version__ >= "1.8":
    import torch_npu


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help="dataset path")
parser.add_argument("--landmarks_type", choices=['2D', '3D'], required=True, help="landmarks type")
parser.add_argument("--steps", type = int, required=False, help="steps")
args = parser.parse_args()


option = {}
option["ACL_OP_COMPILER_CACHE_MODE"] = "enable" # cache功能启用
option["ACL_OP_COMPILER_CACHE_DIR"] = "./my_kernel_meta" # cache所在文件夹
torch.npu.set_option(option)

data_path = args.data_path
CALCULATE_DEVICE = "npu:0"
torch.npu.set_device(CALCULATE_DEVICE)

if args.landmarks_type=='2D':
    landmarks_type=face_alignment.LandmarksType._2D
else:
    landmarks_type=face_alignment.LandmarksType._3D

fa = face_alignment.FaceAlignment(landmarks_type, flip_input=False, device=CALCULATE_DEVICE)
count = 0

val_dataset = ImgDataset(dataset=data_path)
tot_time = AverageMeter('Time', ':6.3f')
torch.npu.synchronize()
end = time.time()
pred = {}
with tqdm(range(len(val_dataset)), desc='Test') as tbar:
    for i, data in enumerate(val_dataset):
        img = data[0]
        preds = fa.get_landmarks_from_image(img)
        torch.npu.synchronize()
        current_batch_time = time.time() - end
        print("sec/step : {}".format(current_batch_time))
        if args.steps:
            count = count + 1
            if count >=args.steps:
                break
        torch.npu.synchronize()
        end = time.time()
        if i > 5:
            tot_time.update(current_batch_time)
        if preds is not None:
            pred[data[1]] = preds
            if args.landmarks_type=='2D':
                plt.imshow(img)
                for detection in preds:
                    plt.scatter(detection[:, 0], detection[:, 1], 2)
                plt.axis('off')
                name = data[1].split('.')[0]
                name = name+'.png'
                saves = os.path.join('./result/images/2D/', name)
                plt.savefig(saves, bbox_inches='tight')
                plt.cla()
            else:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                for detection in preds:
                    ax.scatter3D(detection[:, 0], detection[:, 1], detection[:, 2])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                name = data[1].split('.')[0]
                name = name+'.png'
                saves = os.path.join('./result/images/3D/', name)
                plt.savefig(saves, bbox_inches='tight')
                plt.cla()
        else:
            pred[data[1]] = 'None'
        tbar.update()
if args.landmarks_type=='2D':
    print('Total FPS2D = {:.2f}\t'.format(1 / tot_time.avg))
    np.save('./result/points/2D_npu', pred)
else:
    print('Total FPS3D = {:.2f}\t'.format(1 / tot_time.avg))
    np.save('./result/points/3D_npu', pred)

