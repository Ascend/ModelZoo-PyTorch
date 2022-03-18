# -*- coding: utf-8 -*-
# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
import os
#NPU_CALCULATE_DEVICE = 0
#if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    #NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.001)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=9)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16)
    parser.add_argument(
        '--step_size',
        type=int,
        default=7)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)

    # Overall Dataset settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="data/activitynet_annotations/video_info_new.csv")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="data/activitynet_annotations/anet_anno_action.json")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="data/activitynet_feature_cuhk/")

    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5)

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=400)

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=8)
    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.4)
    parser.add_argument(
        '--soft_nms_low_thres',
        type=float,
        default=0.5)
    parser.add_argument(
        '--soft_nms_high_thres',
        type=float,
        default=0.9)
    parser.add_argument(
        '--result_file',
        type=str,
        default="./output/result_proposal.json")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="./output/evaluation_result.jpg")
    
    parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='hccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('-bm', '--benchmark', default=0, type=int,
                        metavar='N', help='set benchmark status (default: 1,run benchmark)')
                    
    parser.add_argument('--addr', default='10.136.181.115', type=str, help='master addr')
    parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
    parser.add_argument('--device_num', default=-1, type=int,
                        help='device_num')
    parser.add_argument('--device-list', default='', type=str, help='device id list')
    
    # apex
    parser.add_argument('--amp', default=False, action='store_true',
                        help='use amp to train the model')
    parser.add_argument('--loss-scale', default=128., type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt-level', default='O2', type=str,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='number of classes')
                        
    # inference
    parser.add_argument('--input_file', type=str, default='checkpoint/BMN_best.pth.tar', help='verbose option at export.')
    parser.add_argument('--output_file', type=str, default='bmn-bs1.onnx', help='output filepath.')
    parser.add_argument('--infer_batch_size', type=int, default=1, help='batch size to test.')
    parser.add_argument('--opset_version', type=int, default=11, help='opset version at export.')
    parser.add_argument('--verbose', action='store_true', help='verbose option at export.')
    
    args = parser.parse_args()

    return args

