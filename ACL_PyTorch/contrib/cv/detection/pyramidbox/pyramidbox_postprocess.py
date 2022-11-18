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

from __future__ import absolute_import
from __future__ import division

import sys
import os
import torch
import argparse
import numpy as np
import cv2
import os.path as osp
import torch.backends.cudnn as cudnn

from PIL import Image
import scipy.io as sio
from data.config import cfg
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr
from layers.bbox_utils import decode, nms

# parser = argparse.ArgumentParser(description='pyramidbox evaluatuon wider')
# parser.add_argument('--thresh', default=0.05, type=float,
#                     help='Final confidence threshold')
# args = parser.parse_args()

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
List1 = []
List2 = []
root_path = os.getcwd()
def pre_postprocess(i,path):
    listt=[]
    global t
    if i==0:
        path = os.path.join(root_path,path)
    else:
        path = os.path.join(root_path,'result/result22')
    File = os.listdir(path)
    for file in sorted(File):
       Doc = []  #save no-repeated file name
       os.chdir(os.path.join(path, file))
       cur_path = os.getcwd()
       doc = os.listdir(cur_path)
       for document in sorted(doc):
           Doc.append(document[0:-6])  #grip end
           Doc = list(set(Doc))     #grip repeated element
       for ff in sorted(Doc):    #deal after sorting
           txt_file = np.fromfile(f'{path}/{file}/{ff}_1.bin', dtype=np.float16)
           output = torch.tensor(txt_file.reshape(-1,1000,5))
           listt.append(output)
    return listt
 
def detect_face(img, counter,i):
    h, w = img.shape[0], img.shape[1]
    min_side = 1280 
    scale = max(w, h) / float(min_side)
    if i==0:
        detections = List1[counter].data
    else:
        detections = List2[counter].data
    detections = detections.cpu().numpy()
    det_conf = detections[0, :, 0]
    det_xmin = 1280 * detections[0, :, 1] * scale  #x1
    det_ymin = 1280 * detections[0, :, 2] * scale  #y1
    det_xmax = 1280 * detections[0, :, 3] * scale  #x2
    det_ymax = 1280 * detections[0, :, 4] * scale  #y2
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))
    keep_index = np.where(det[:, 4] >= 0.05)[0]
    det = det[keep_index, :]
    return det

def flip_test(image,counter,i):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f,counter,1)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t

def multi_scale_test(image, max_im_shrink,counter,i):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image,counter,i)
    index = np.where(np.maximum(
        det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (
        st + max_im_shrink) / 2
    det_b = detect_face(image,counter,i)
 
        
    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b
    
def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        
        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
        dets = dets[0:750, :]
    return dets
    
if __name__ == '__main__':
    mat_path = os.path.abspath(sys.argv[1]) #mat path    './evaluate/ground_truth/wider_face_val.mat'
    imgs_path = os.path.abspath(sys.argv[2]) #image path    './images'
    save_path = os.path.abspath(sys.argv[3]) #save path   './output_0.01/widerface/'
    path1 = os.path.abspath(sys.argv[4]) #first data ---> result    './result1/result1'
    path2 = os.path.abspath(sys.argv[5]) #second data ---> result    './result2/result2'
    wider_face = sio.loadmat('./evaluate/ground_truth/wider_face_val.mat')
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face
    # imgs_path = root_path+'/images'
    # save_path = root_path+'/output_1000'
    counter = 0
    if use_cuda:
        cudnn.benckmark = True
    # path1 = './result/result11'
    # path2 = './result/result2/result2'
    List1 = pre_postprocess(0,path1)
    List2 = pre_postprocess(1,path2)
    print(List1)
    print(len(List1))
    print('-----------------------------------------------')
    print(len(List2))
    print(List2)    
    i=0
    for index, event in enumerate(sorted(event_list)):
        filelist = file_list[index][0]
        path = os.path.join(save_path, str(event[0][0]))
        if not os.path.exists(path):
            os.makedirs(path)
        i = i+1
        for num, file in enumerate(sorted(filelist)):
            im_name = file[0][0]        
            print(im_name)  
            in_file = os.path.join(imgs_path, event[0][0], str(im_name[:]) + '.jpg')
            img = Image.open(in_file)
            if img.mode == 'L':
                img = img.convert('RGB')
            img = np.array(img)
            max_im_shrink = np.sqrt(
                1700 * 1000 / (img.shape[0] * img.shape[1]))
            shrink = max_im_shrink if max_im_shrink < 1 else 1
            counter += 1
            det0 = detect_face(img,counter-1,0)

            det1 = flip_test(img,counter-1,1)    # flip test
            [det2, det3] = multi_scale_test( img, max_im_shrink,counter-1,0)

            det = np.row_stack((det0, det1, det2, det3))
            if det.shape[0] ==1:
              dets =det
            else:
              dets = bbox_vote(det)

            fout = open(osp.join(save_path, str(event[0][
                        0]), im_name + '.txt'), 'w')
            fout.write('{:s}\n'.format(str(event[0][0]) + '/' + im_name + '.jpg'))
            fout.write('{:d}\n'.format(dets.shape[0]))
            for i in range(dets.shape[0]):
                xmin = dets[i][0]
                ymin = dets[i][1]
                xmax = dets[i][2]
                ymax = dets[i][3]
                score = dets[i][4]
                fout.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                           format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))