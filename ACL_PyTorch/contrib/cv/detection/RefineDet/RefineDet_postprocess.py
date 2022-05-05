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
import sys
sys.path.append('./RefineDet.PyTorch')
from eval_utils import *
from data import VOCAnnotationTransform, VOCDetection, BaseTransform 
from layers.functions.detection_refinedet import Detect_RefineDet
import torch
import numpy as np
import pickle
import os
import time
import argparse

def test_net(dataset, det_nms, result_path, set_type='test'):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')
    detection_list = []
    h_list, w_list = dataset.get_h_w_list()
    prior_data = torch.from_numpy(np.loadtxt('prior_data.txt', dtype=np.float32).reshape(6375, 4))
    for i in range(num_images):
        start = time.time()
        res_ls = []
        for j in range(1, 5):
            bin_path = os.path.join(result_path, '%07d_%d.bin'%(i+1,j))
            out1 = np.fromfile(bin_path ,dtype=np.float32)
            res_ls.append(out1)
        # 参数位置对调
        odm_loc_data, odm_conf_data,arm_loc_data,arm_conf_data = res_ls
        arm_loc_data = torch.from_numpy(arm_loc_data.reshape(1, 6375, 4))
        arm_conf_data = torch.from_numpy(arm_conf_data.reshape(1, 6375, 2))
        odm_loc_data = torch.from_numpy(odm_loc_data.reshape(1, 6375, 4))
        odm_conf_data = torch.from_numpy(odm_conf_data.reshape(1, 6375, 21))
        detections = det_nms.forward(arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data)
        detection_list.append(detections)
        end = time.time() 
        print('%d / %d spend time: %.3fs'%(i+1,num_images,(end-start)))
    strat_time = time.time()
    detections = torch.cat(detection_list, dim=0)
    for idx in range(detections.size(0)):
        h, w = h_list[idx], w_list[idx]
        for j in range(1, detections.size(1)):
            dets = detections[idx, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                scores[:, np.newaxis])).astype(np.float32,
                                                                copy=False)       
            all_boxes[j][idx] = cls_dets 
    end_time = time.time()
    print('spend time: %.3fs'%(end_time-strat_time))
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    mAp = evaluate_detections(all_boxes, output_dir, dataset.ids)
    return mAp

if __name__ == '__main__':
    num_classes = len(labelmap) + 1                      
    dataset_mean = (104, 117, 123)
    set_type = 'test'
    dataset = VOCDetection(root = voc_root,
                           image_sets=[('2007', set_type)],
                           transform=BaseTransform(320, dataset_mean),
                           target_transform=VOCAnnotationTransform(),
                           dataset_name='VOC07test')
    det_nms = Detect_RefineDet(21, 320, 0, 1000, 0.01, 0.45, 0.01, 500)
    mAp = test_net(dataset, det_nms, result_path, set_type='test')
    with open('acc.txt', 'w') as f:
        f.write('mAp: ' + str(mAp))
