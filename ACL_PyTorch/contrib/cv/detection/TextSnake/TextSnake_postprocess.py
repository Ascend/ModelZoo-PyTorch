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
import os
import time
import cv2
import torch.utils.data as data
import subprocess
import sys
from tqdm import tqdm
sys.path.append('./TextSnake.pytorch')
from util.detection import TextDetector
from util.augmentation import BaseTransform
from util.option import BaseOptions
from util.config import config as cfg, update_config, print_config
from dataset.total_text import TotalText
from util.misc import to_device, mkdirs, rescale_result
from network.textnet import TextNet

class Detector(TextDetector):

    def detect(self, image, data):
        output = torch.from_numpy(data)
        image = image[0].data.cpu().numpy()
        tr_pred = output[0, 0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred = output[0, 2:4].softmax(dim=0).data.cpu().numpy()
        sin_pred = output[0, 4].data.cpu().numpy()
        cos_pred = output[0, 5].data.cpu().numpy()
        radii_pred = output[0, 6].data.cpu().numpy()
        contours = self.detect_contours(image, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred)  # (n_tcl, 3)
        output = {
            'image': image,
            'tr': tr_pred,
            'tcl': tcl_pred,
            'sin': sin_pred,
            'cos': cos_pred,
            'radii': radii_pred
        }
        return contours, output

def write_to_file(contours, file_path):
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 1], cont[:, 0]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')

def inference(detector, test_loader, output_dir, image_list):

    total_time = 0.
    total_data = []
    path = './result/dumpOutput_device0'

    print('read output file')

    for file_name in tqdm(image_list):
        file_prefix = file_name.split('.')[0]
        file = file_prefix + "_1.txt"
        file_path = os.path.join(path, file)
        data = np.loadtxt(file_path).reshape((1, 7, 512, 512))
        total_data.append(data)

    for i, (image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(test_loader):
        start = time.time()
        idx = 0 # test mode can only run with batch_size == 1

        # file_name = image_list[i]
        # print(file_name)
        # print('detect {} / {} images: {}'.format(i + 1, len(test_loader), meta['image_id'][idx]))
        # file_prefix = file_name.split('.')[0]
        # idex = int(file_prefix[3:]) - 1
        # print(idex)
        # file = file_prefix + "_1.txt"
        
        # file_path = os.path.join(path, file)
        # data = np.ones((1, 7, 512, 512))
        data = total_data[i]
        contours, output = detector.detect(image, data)

        # end = time.time()
        # total_time += end - start
        # fps = (i + 1) / total_time
        print('detect {} / {} images: {}'.format(i + 1, len(test_loader), meta['image_id'][idx]))

        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        img_show, contours = rescale_result(img_show, contours, H, W)

        mkdirs(output_dir)
        write_to_file(contours, os.path.join(output_dir, meta['image_id'][idx].replace('jpg', 'txt')))


def main():
    
    testset =  TotalText(
        data_root='./data/total-text',
        ignore_list=None,
        is_training=False,
        transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
    )

    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    model = TextNet(is_training=False, backbone=cfg.net)
    detector = Detector(model, tr_thresh=cfg.tr_thresh, tcl_thresh=cfg.tcl_thresh)

    print('Start testing TextSnake.')
    output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    inference(detector, test_loader, output_dir, testset.image_list)

    print('Computing DetEval in {}/{}'.format(cfg.output_dir, cfg.exp_name))
    subprocess.call(['python3.7', './TextSnake.pytorch/dataset/total_text/Evaluation_Protocol/Python_scripts/Deteval.py', args.exp_name, '--tr', '0.7', '--tp', '0.6'])
    subprocess.call(['python3.7', './TextSnake.pytorch/dataset/total_text/Evaluation_Protocol/Python_scripts/Deteval.py', args.exp_name, '--tr', '0.8', '--tp', '0.4'])
    print('End.')

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    # print_config(cfg)

    main()
