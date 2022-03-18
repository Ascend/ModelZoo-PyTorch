#coding=gbk
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
from __future__ import print_function
import os
import warnings
warnings.filterwarnings('ignore')
import torch
import pickle
import argparse
import numpy as np
from m2det import build_net
from utils.timer import Timer
import torch.backends.cudnn as cudnn
from layers.functions import Detect,PriorBox
from data import BaseTransform
from configs.CC import Config
from tqdm import tqdm
from utils.core import *
#NPU修改开始
import torch.npu
#NPU修改结束
parser = argparse.ArgumentParser(description='M2Det Testing')
parser.add_argument('-c', '--config', default='configs/m2det512_vgg.py', type=str)
parser.add_argument('-d', '--dataset', default='COCO', help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default=None, type=str, help='Trained state_dict file path to open')
parser.add_argument('--test', action='store_true', help='to submit a test file')
#NPU参数开始
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str,
                        help='device id list')
parser.add_argument('--device', default='npu', type=str, help='npu or cpu')
parser.add_argument('--addr', default='127.0.0.1', type=str,
                        help='master addr')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
#NPU参数结束
args = parser.parse_args()


def test_net(save_folder, net, detector, cuda, testset, transform, args, max_per_image=300, thresh=0.005):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    num_images = len(testset)
    print_info('=> Total {} images to test.'.format(num_images),['yellow','bold'])
    num_classes = cfg.model.m2det_config.num_classes
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')
    tot_detect_time, tot_nms_time = 0, 0
    print_info('Begin to evaluate',['yellow','bold'])
    for i in tqdm(range(num_images)):
        img = testset.pull_image(i)
        # step1: CNN detection
        _t['im_detect'].tic()
        boxes, scores = image_forward(img, net, cuda, priors, detector, transform)
        detect_time = _t['im_detect'].toc()
        # step2: Post-process: NMS
        _t['misc'].tic()
        nms_process(num_classes, i, scores, boxes, cfg, thresh, all_boxes, max_per_image)
        nms_time = _t['misc'].toc()

        tot_detect_time += detect_time if i > 0 else 0
        tot_nms_time += nms_time if i > 0 else 0

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print_info('===> Evaluating detections',['yellow','bold'])
    testset.evaluate_detections(all_boxes, save_folder)
    print_info('Detect time per image: {:.3f}s'.format(tot_detect_time / (num_images - 1)))
    print_info('Nms time per image: {:.3f}s'.format(tot_nms_time / (num_images - 1)))
    print_info('Total time per image: {:.3f}s'.format((tot_detect_time + tot_nms_time) / (num_images - 1)))
    print_info('FPS: {:.3f} fps'.format((num_images - 1) / (tot_detect_time + tot_nms_time)))

if __name__ == '__main__':
    print_info('----------------------------------------------------------------------\n'
               '|                       M2Det Evaluation Program                     |\n'
               '----------------------------------------------------------------------', ['yellow','bold'])
    """
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    elif args.device.startswith('npu'):
        torch.npu.set_device(args.device)
    else:
        pass
    """
    calculate_device = 'npu:{}'.format(args.device_list)
    print("device",calculate_device)
    torch.npu.set_device(calculate_device)
    
    global cfg
    cfg = Config.fromfile(args.config)
    if not os.path.exists(cfg.test_cfg.save_folder):
        os.mkdir(cfg.test_cfg.save_folder)
    anchor_config = anchors(cfg)
    print_info('The Anchor info: \n{}'.format(anchor_config))
    priorbox = PriorBox(anchor_config)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.npu()
    
    net = build_net('test',
                    size = cfg.model.input_size,
                    config = cfg.model.m2det_config)
    init_net(net, cfg, args.trained_model)
    print_info('===> Finished constructing and loading model',['yellow','bold'])
    net.eval()
    _set = 'eval_sets' if not args.test else 'test_sets'
    testset = get_dataloader(cfg, args.dataset, _set)
    net = net.npu()
    cudnn.benchmark = True

    detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)
    save_folder = os.path.join(cfg.test_cfg.save_folder, args.dataset)
    _preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
    test_net(save_folder, 
             net, 
             detector, 
             cfg.test_cfg.cuda, 
             testset, 
             transform = _preprocess,
             args = args, 
             max_per_image = cfg.test_cfg.topk, 
             thresh = cfg.test_cfg.score_threshold,
             )
