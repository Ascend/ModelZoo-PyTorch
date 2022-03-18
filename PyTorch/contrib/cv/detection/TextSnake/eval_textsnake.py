# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
import os
import time
import cv2
import numpy as np
import torch
import subprocess
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from dataset.total_text import TotalText
from network.textnet import TextNet
from util.detection import TextDetector
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config, print_config
from util.option import BaseOptions
from util.visualize import visualize_detection
from util.misc import to_device, mkdirs, rescale_result

def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 1], cont[:, 0]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')


def inference(detector, test_loader, output_dir):

    total_time = 0.

    for i, (image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(test_loader):

        image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
            image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

        #torch.cuda.synchronize()
        start = time.time()

        idx = 0 # test mode can only run with batch_size == 1

        # get detection result
        contours, output = detector.detect(image)

        #torch.cuda.synchronize()
        end = time.time()
        total_time += end - start
        fps = (i + 1) / total_time
        print('detect {} / {} images: {}. ({:.2f} fps)'.format(i + 1, len(test_loader), meta['image_id'][idx], fps))

        # visualization
        tr_pred, tcl_pred = output['tr'], output['tcl']
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        pred_vis = visualize_detection(img_show, contours, tr_pred[1], tcl_pred[1])
        gt_contour = []
        for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
            if n_annot.item() > 0:
                gt_contour.append(annot[:n_annot].int().cpu().numpy())
        gt_vis = visualize_detection(img_show, gt_contour, tr_mask[idx].cpu().numpy(), tcl_mask[idx].cpu().numpy())
        im_vis = np.concatenate([pred_vis, gt_vis], axis=0)
        path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name), meta['image_id'][idx])
        cv2.imwrite(path, im_vis)

        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        img_show, contours = rescale_result(img_show, contours, H, W)

        # write to file
        mkdirs(output_dir)
        write_to_file(contours, os.path.join(output_dir, meta['image_id'][idx].replace('jpg', 'txt')))

def main():

    testset = TotalText(
        data_root='data/total-text',
        ignore_list=None,
        is_training=False,
        transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
    )
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TextNet(is_training=False, backbone=cfg.net)
    if cfg.device == 'npu':
        loc = 'npu:{}'.format(cfg.gpu)
        torch.npu.set_device(cfg.gpu)
        model = model.to(loc)

    #model_path = os.path.join(cfg.save_dir, cfg.exp_name, \
    #          'textsnake_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
    model_path = os.path.join(cfg.save_dir, cfg.exp_name, \
              'textsnake_{}.pth'.format(cfg.checkepoch))
    model.load_model(model_path)

    # copy to cuda
    #model = model.to(cfg.device)
    #if cfg.cuda:
    #    cudnn.benchmark = True
    detector = TextDetector(model, tr_thresh=cfg.tr_thresh, tcl_thresh=cfg.tcl_thresh)

    print('Start testing TextSnake.')
    output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    inference(detector, test_loader, output_dir)

    # compute DetEval
    print('Computing DetEval in {}/{}'.format(cfg.output_dir, cfg.exp_name))
    subprocess.call(['python3.7', 'dataset/total_text/Evaluation_Protocol/Python_scripts/Deteval.py', args.exp_name, '--tr', '0.7', '--tp', '0.6'])
    #subprocess.call(['python', 'dataset/total_text/Evaluation_Protocol/Python_scripts/Deteval.py', args.exp_name, '--tr', '0.8', '--tp', '0.4'])
    print('End.')


if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    vis_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)
    # main
    main()