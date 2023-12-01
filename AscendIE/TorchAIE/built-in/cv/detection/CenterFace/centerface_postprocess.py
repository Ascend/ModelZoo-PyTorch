# Copyright 2023 Huawei Technologies Co., Ltd
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

# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import os
import sys
import cv2
import scipy.io as sio

from models import decode as dc
from models.utils import _gather_feat, _tranpose_and_gather_feat
from utils import image as img
from utils.post_process import multi_pose_post_process
from opts_pose import opts
sys.path.insert(0, '..')
from detectors.detector_factory import detector_factory
from datasets.dataset_factory import get_dataset

def preprocess(Path):
    root_path = os.getcwd()
    opt = opts().parse('--task {} --load_model {}'.format('multi_pose',
                                                          os.path.join(root_path, 'model_best.pth')).split(' '))
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    in_files = os.listdir(Path)
    Meta = []
    for file in sorted(in_files):
        os.chdir(os.path.join(Path, file))
        cur_path = os.getcwd()
        doc = os.listdir(cur_path)
        for document in sorted(doc):
            image = cv2.imread(os.path.join(cur_path, document))
            if document == 'output':
                break
            if not document.endswith('jpg'):
                continue
            for scale in opt.test_scales:
                images, meta = detector.pre_process(image, scale, meta=None)
                Meta.append(meta)
    return Meta


def post_process(dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'])
    for j in range(1, 2):
        dets[0][j] = np.array(
            dets[0][j], dtype=np.float32).reshape(-1, 15)             # 关键点数+5=15
        dets[0][j][:, :4] /= scale
        dets[0][j][:, 5:] /= scale
    return dets[0]


def merge_outputs(detections):
    results = {}
    results[1] = np.concatenate(
        [detection[1] for detection in detections], axis=0).astype(np.float32)
    results[1] = results[1].tolist()
    return results


def pre_postprocess():
    List = []
    root_path = os.getcwd()
    path = os.path.join(root_path, './result/bs1')
    File = os.listdir(path)
    for file in sorted(File):
        Doc = []  # save no-repeated file name
        os.chdir(os.path.join(path, file))
        cur_path = os.getcwd()
        doc = os.listdir(cur_path)
        for document in sorted(doc):
            Doc.append(document[0:-6])  # grip end
            Doc = list(set(Doc))  # grip repeated element
        for ff in sorted(Doc):  # deal after sorting
            dist = {}
            if ff == 'kerne':
                break
            for i in range(4):  # one image ----->four bin
                txt_file = np.fromfile(
                    f'../../../result/bs1/{file}/{ff}_{i}.bin', dtype=np.float16).astype(np.float32)
                if i == 0:
                    dist['hm'] = torch.tensor(
                        txt_file.reshape(-1, 1, 200, 200))
                if i == 1:
                    dist['wh'] = torch.tensor(
                        txt_file.reshape(-1, 2, 200, 200))
                if i == 2:
                    dist['hm_offset'] = torch.tensor(
                        txt_file.reshape(-1, 2, 200, 200))
                if i == 3:
                    dist['landmarks'] = torch.tensor(
                        txt_file.reshape(-1, 10, 200, 200))
            List.append(dist)
    os.chdir(root_path)
    return List


def run(Path):
    List = pre_postprocess()
    Meta = preprocess(Path)
    print('List:', len(List))
    print('Meta:', len(Meta))
    Results = []
    from tqdm import tqdm
    for i in tqdm(range(len(List))):
        detections = []
        reg = List[i]['hm_offset']
        dets = dc.centerface_decode(
            List[i]['hm'], List[i]['wh'], List[i]['landmarks'],
            reg=reg, K=200)
        dets = post_process(dets, Meta[i])
        detections.append(dets)
        results = merge_outputs(detections)
        Results.append(results)
    return Results


if __name__ == "__main__":
    root_path = os.getcwd()
    Path = os.path.join(root_path, '../../WIDER_val/images')
    wider_face_mat = sio.loadmat(root_path + '/../../evaluate/ground_truth/wider_face_val.mat')
    event_list = wider_face_mat['event_list']  # directory
    file_list = wider_face_mat['file_list']  # file
    save_path = os.path.join(root_path, '../output/widerface/')
    results = run(Path)  # all  data
    i = 0  # iteration
    for index, event in enumerate(sorted(event_list)):
        file_list_item = file_list[index][0]
        im_dir = event[0][0]
        if not os.path.exists(save_path + im_dir):
            os.makedirs(save_path + im_dir)
        for num, file in enumerate(sorted(file_list_item)):
            im_name = file[0][0]
            zip_name = '%s/%s.jpg' % (im_dir, im_name)
            img_path = os.path.join(Path, zip_name)
            dets = results[i]
            f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            f.write('{:d}\n'.format(len(dets)))
            for b in dets[1]:
                x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(
                    x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            f.close()
            print(i)
            i = i+1
