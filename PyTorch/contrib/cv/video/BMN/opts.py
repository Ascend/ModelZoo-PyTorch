# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import argparse
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--finetune',
        type=int,
        default=0)
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
    parser.add_argument(
        '--pth_path',
        type=str,
        default='./checkpoint/BMN_best.pth.tar')
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
        default=10)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8)
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
        '--data_path',
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
    
    parser.add_argument("--is_distributed", type=int, default=0, help='choose ddp or not')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--DeviceID', type=str, default="0")
    parser.add_argument('--world_size', type=int, default=1)
    
    # apex

    parser.add_argument('--loss-scale', default=128., type=float,
                        help='loss scale using in amp, default -1 means dynamic')
    parser.add_argument('--opt-level', default='O1', type=str,
                        help='loss scale using in amp, default -1 means dynamic')


    args = parser.parse_args()

    return args

