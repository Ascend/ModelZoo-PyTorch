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
from __future__ import print_function
import torch.nn.functional as F
import sys
sys.path.append('./RFCN-pytorch.1.0')
import _init_paths
import os

import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.rfcn.resnet_atrous import resnet
import pdb
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='test the accuracy of RFCN')

    parser.add_argument("--image_folder_path", dest="file_path", default="./RFCN-pytorch.1.0/data/VOCdevkit2007/VOC2007/JPEGImages/",help='image of dataset')
    parser.add_argument("--input",dest="input", default="./result/dumpOutput_device0/")
    parser.add_argument("--output",dest="output", default="./output")
    parser.add_argument("--net_input_width", default=1344)
    parser.add_argument("--net_input_height", default=1344)
    parser.add_argument('--dataset', dest='dataset',help='training dataset',default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file',default='cfgs/res16.yml', type=str)
    parser.add_argument('--net', dest='net',help='vgg16, res50, res101, res152',default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',help='set config keys', default=None,nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',help='directory to load models', default="models", type=str)
    parser.add_argument('--ls', dest='large_scale',help='whether use large imag scale',action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',help='whether perform class_agnostic bbox regression',action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',help='which part of model to parallel, 0: all, 1: model before roi pooling',default=0, type=int)
    parser.add_argument('--bs', dest='batch_size',help='batch_size', default=1, type=int)
    parser.add_argument('--vis', dest='vis', help='visualization mode',action='store_true')
    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    np.random.seed(cfg.RNG_SEED)

    args.imdbval_name = "voc_2007_test"
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    args.cfg_file = "./RFCN-pytorch.1.0/cfgs/{}_ls.yml".format(args.net) if args.large_scale else "./RFCN-pytorch.1.0/cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    #save_name = 'RFCN'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    #output_dir = get_output_dir(imdb, save_name)
    output=args.output
    if not os.path.exists(output):
        os.makedirs(output)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=False, normalize = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    # _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output, 'detections.pkl')
    # fasterRCNN.eval()
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
    dataset.resize_batch()
    npu_result = args.input
    with open("./RFCN-pytorch.1.0/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt") as f:
        imglist = [x.strip() for x in f.readlines()]
    num_images = len(imglist)
    for i in range(num_images):
        data = next(data_iter)
        pad_value = 0
        batch_shape = (3, 1344, 1344)
        padding_size = [0, batch_shape[-1] - data[0].shape[-1],
                        0, batch_shape[-2] - data[0].shape[-2]]
        #data[0] = F.pad(data[0], padding_size, value=pad_value)
        #im_data.resize_(data[0].size()).copy_(data[0])
        # print(im_data.size())
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])
        det_tic = time.time()

        def read_data(data_path, input_shape=None):
            if data_path.endswith('.bin'):
                data = np.fromfile(data_path, dtype=np.float32)
                data = data.reshape(input_shape)
            elif data_path.endswith('.npy'):
                data = np.load(data_path)
            return data

        rois = torch.from_numpy(
            read_data(npu_result+'/'+'{}_0.bin'.format(imglist[i]), [1, 300, 5]))
        cls_prob = torch.from_numpy(
            read_data(npu_result+'/'+'{}_1.bin'.format(imglist[i]), [1, 300, 21]))
        bbox_pred = torch.from_numpy(
            read_data(npu_result+'/'+'{}_2.bin'.format(imglist[i]), [1, 300, 84]))
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        box_deltas = box_deltas.view(args.batch_size, -1, 4 * len(imdb.classes))
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        pred_boxes /= data[1][0][2]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            cv2.imwrite('result.png', im2show)
            pdb.set_trace()
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output)

    end = time.time()
    print("test time: %0.4fs" % (end - start))