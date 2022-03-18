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
import sys
sys.path.append('./RFCN-pytorch.1.0')
import _init_paths

import numpy as np
import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
#from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
# from model.roi_layers import nms
#from model.rpn.bbox_transform import bbox_transform_inv
#from model.utils.net_utils import save_net, load_net, vis_detections
from model.rfcn.resnet_atrous import resnet
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    parser = argparse.ArgumentParser(description='product onnx')
    parser.add_argument('--dataset', dest='dataset',help='training dataset',default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',help='optional config file',default='RFCN-pytorch.1.0/cfgs/res16.yml', type=str)
    parser.add_argument('--net', dest='net',help='vgg16, res50, res101, res152', default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',help='set config keys', default=None,nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',help='directory to load models', default="models",type=str)
    parser.add_argument('--ls', dest='large_scale',help='whether use large imag scale',action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',help='whether perform class_agnostic bbox regression',action='store_true')
    parser.add_argument('--bs', dest='batch_size',help='batch_size',default=1, type=int)
    parser.add_argument('--vis', dest='vis',help='visualization mode',action='store_true')
    parser.add_argument('--input', dest='input')
    parser.add_argument('--output', dest='output')
    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':
    args = parse_args()
    input_file = args.input
    output_file = args.output
    load_name = input_file
    
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

   
    # initilize the network here.
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    
    checkpoint = torch.load(load_name,map_location='cpu')
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=False, normalize = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)
    data_iter = iter(dataloader)

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
    dataset.resize_batch()

    data = next(data_iter)
    pad_value = 0
    batch_shape = (3,1344,1344)
    padding_size = [0, batch_shape[-1] - data[0].shape[-1],
                    0, batch_shape[-2] - data[0].shape[-2]]
    data[0] = torch.nn.functional.pad(data[0], padding_size, value=pad_value)
    with torch.no_grad():
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])

    torch.onnx.export(fasterRCNN, (im_data, im_info, gt_boxes, num_boxes), output_file,
            input_names=['im_data', 'im_info', 'gt_boxes', 'num_boxes'], output_names=['rois', 'cls_prob', 'bbox_pred',], opset_version=11, enable_onnx_checker=False, verbose=True, do_constant_folding=False)
