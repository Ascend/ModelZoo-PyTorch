'''
# Copyright 2020 Huawei Technologies Co., Ltd
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
'''

import os
import sys
sys.path.insert(0, './M2Det')
import numpy as np
import argparse
import cv2
import warnings
warnings.filterwarnings('ignore')
import torch
import pickle
from utils.timer import Timer
from layers.functions import Detect, PriorBox
from data import BaseTransform
from configs.CC import Config
from utils.core import set_train_log, anchors, get_dataloader, nms_process, print_info
from data import COCODetection, VOCDetection, detection_collate, preproc
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M2Det Postprocess')
    parser.add_argument("--bin_data_path", default="./result/dumpOutput_device0")
    parser.add_argument("--test_annotation", default="./coco_images.info")
    parser.add_argument("--det_results_path", default="./detection-results/")
    parser.add_argument("--net_out_num", type=int, default=2)
    parser.add_argument("--net_input_width", type=int, default=512)
    parser.add_argument("--net_input_height", type=int, default=512)
    parser.add_argument("--prob_thres", default=0.1)
    parser.add_argument('-c', '--config', default='M2Det/configs/m2det512_vgg.py', type=str)
    parser.add_argument('-d', '--dataset', default='COCO', help='VOC or COCO version')
    parser.add_argument('-m', '--trained_model', default=None, type=str, help='Trained state_dict file path to open')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set which type of device used. Support cuda:0(device_id), npu:0(device_id).')
    parser.add_argument('--test', action='store_true', help='to submit a test file')
    parser.add_argument('--COCO_imgs', default="~/data/coco/images", help='COCO images root')
    parser.add_argument('--COCO_anns', default="~/data/coco/annotations", help='COCO annotations root')
    args = parser.parse_args()
    
    logr = set_train_log()
    # read bin file for generate predict result
    bin_path = args.bin_data_path

    save_folder = os.path.join(args.det_results_path, args.dataset)
    det_file = os.path.join(save_folder, 'detections.pkl')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    cfg = Config.fromfile(args.config)
    num_classes = cfg.model.m2det_config.num_classes
    thresh = cfg.test_cfg.score_threshold
    max_per_image = cfg.test_cfg.topk
    anchor_config = anchors(cfg)
    print_info('The Anchor info: \n{}'.format(anchor_config))
    priorbox = PriorBox(anchor_config)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(args.device)
    detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)
    
    _set = 'eval_sets' if not args.test else 'test_sets'
    testset = get_dataloader(args, cfg, args.dataset, _set)

    # generate dict according to annotation file for query resolution
    # load width and height of input images
    img_size_dict = dict()
    with open(args.test_annotation)as f:
        for line in f.readlines():
            temp = line.split(" ")
            img_file_path = temp[1]
            img_name = temp[1].split("/")[-1].split(".")[0]
            img_width = int(temp[2])
            img_height = int(temp[3])
            img_size_dict[img_name] = (img_width, img_height, img_file_path)

    total_img = set([name[:name.rfind('_')] for name in os.listdir(bin_path) if "bin" in name])
    num_images = len(total_img)
    print('num_images:{}'.format(num_images))

    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    _t = {'im_detect': Timer(), 'misc': Timer()}
    tot_detect_time, tot_nms_time = 0, 0
    
    cnt = 0
    for bin_file in sorted(total_img):
        print('i:{}'.format(cnt))
        
        path_base = os.path.join(bin_path, bin_file)
        # load all detected output tensor
        _t['im_detect'].tic()
        for num in range(1, args.net_out_num + 1):
            if os.path.exists(path_base + "_" + str(num) + ".bin"):
                if num == 1:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")#scores
                    score = np.reshape(buf, [32760, 81])
                    
                elif num == 2:
                    buf = np.fromfile(path_base + "_" + str(num) + ".bin", dtype="float32")#int64 boxes
                    box = np.reshape(buf, [1, 32760, 4])
    
            else:
                print("[ERROR] file not exist", path_base + "_" + str(num) + ".bin")
        
        out = (torch.from_numpy(box), torch.from_numpy(score))
        boxes, scores = detector.forward(out, priors)
        
        current_img_size = img_size_dict[bin_file]
        w = current_img_size[0]
        h = current_img_size[1]
        scale = torch.Tensor([w, h, w, h])
        boxes = (boxes[0] * scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        detect_time = _t['im_detect'].toc()
        _t['misc'].tic()
        nms_process(num_classes, cnt, scores, boxes, cfg, thresh, all_boxes, max_per_image)
        nms_time = _t['misc'].toc()
        
        tot_detect_time += detect_time if cnt > 0 else 0
        tot_nms_time += nms_time if cnt > 0 else 0
    
        logr.info('Times:{}||scale:{}||boxes:{}||tot_detect_time:{}||tot_nms_time:{}'\
            .format(cnt, scale, boxes, tot_detect_time, tot_nms_time))
        cnt = cnt + 1
        
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print_info('===> Evaluating detections', ['yellow', 'bold'])
    testset.evaluate_detections(all_boxes, save_folder)
    print_info('Detect time per image: {:.3f}s'.format(tot_detect_time / (num_images - 1)))
    print_info('Nms time per image: {:.3f}s'.format(tot_nms_time / (num_images - 1)))
    print_info('Total time per image: {:.3f}s'.format((tot_detect_time + tot_nms_time) / (num_images - 1)))
    print_info('FPS: {:.3f} fps'.format((num_images - 1) / (tot_detect_time + tot_nms_time)))
    logr.info('Detect time per image: {:.3f}s'.format(tot_detect_time / (num_images - 1)))
    logr.info('Nms time per image: {:.3f}s'.format(tot_nms_time / (num_images - 1)))
    logr.info('Total time per image: {:.3f}s'.format((tot_detect_time + tot_nms_time) / (num_images - 1)))
    logr.info('FPS: {:.3f} fps'.format((num_images - 1) / (tot_detect_time + tot_nms_time)))
    logr.info('End...')
    print('End')
    