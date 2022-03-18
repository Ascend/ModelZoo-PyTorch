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

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
sys.path.append(r"./3DMPPE_ROOTNET_RELEASE")
from data.dataset import DatasetLoader
from data.MuPoTS.MuPoTS import MuPoTS

def preprocess(inference_batch_size, save_path_imge, save_path_cam, img_path, ann_path):
    print('preprocessing')
    testset = MuPoTS('test', img_path, ann_path)

    testset_loader = DatasetLoader(testset, False, transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]))
    batch_generator = DataLoader(dataset=testset_loader, batch_size=inference_batch_size,
                                 shuffle=False, num_workers=8, pin_memory=True)

    if not os.path.exists(save_path_imge):
        os.makedirs(save_path_imge)
    if not os.path.exists(save_path_cam):
        os.makedirs(save_path_cam)
    cid = 0
    with torch.no_grad():
        for itr, (input_img, cam_param) in enumerate(tqdm(batch_generator)):
            if(len(input_img) == inference_batch_size):
                path_bin_image = str(save_path_imge) + '/' + str(cid) + '.bin'
                path_bin_cam = str(save_path_cam) + '/' + str(cid) + '.bin'
                cid = cid + 1
                input_img.cpu().numpy().tofile(path_bin_image)
                cam_param.cpu().numpy().tofile(path_bin_cam)
    print('preprocess finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default='MuPoTS/MultiPersonTestSet',
                        type=Path, help='Directory path of videos')
    parser.add_argument('--ann_path', default='MuPoTS/MuPoTS-3D.json', type=Path, help='Annotation file path')
    parser.add_argument('--inference_batch_size', default=1, type=int, help='Batch Size for inference. 0 means this is the same as batch_size.')
    parser.add_argument('--save_path_image', default='0data_imge_bs1', type=Path, help='Directory path of binary output data')
    parser.add_argument('--save_path_cam', default='0data_cam_bs1', type=Path, help='Directory path of binary output data')
    opt = parser.parse_args()
    preprocess(opt.inference_batch_size, opt.save_path_image, opt.save_path_cam, opt.img_path, opt.ann_path)