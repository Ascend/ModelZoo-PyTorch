# Copyright 2022 Huawei Technologies Co., Ltd
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

import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')

    # Overall Dataset settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="./BMN-Boundary-Matching-Network/data/activitynet_annotations/video_info_new.csv")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="./BMN-Boundary-Matching-Network/data/activitynet_annotations/anet_anno_action.json")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="./BMN-Boundary-Matching-Network/data/activitynet_feature_cuhk/")

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
    parser.add_argument(
        '--ground_truth_file',
        type=str,
        default="./BMN-Boundary-Matching-Network/Evaluation/data/activity_net_1_3_new.json")
    
    # tar2onnx
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='checkpoint/BMN_best.pth.tar', 
        help='verbose option at export.'
        )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='bmn_bs1.onnx', 
        help='output filepath.'
        )
    parser.add_argument(
        '--infer_batch_size', 
        type=int, 
        default=1, 
        help='batch size to test.'
        )
    parser.add_argument(
        '--opset_version', 
        type=int, 
        default=11, 
        help='opset version at export.'
        )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='verbose option at export.'
        )

    # preprocess
    parser.add_argument(
        '--data_dir', 
        type=str, 
        help='path to dataset.'
        )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='./prep_bin', 
        help='binary files of preprocessed data.'
        )

    # inference
    parser.add_argument(
        '--result_dir', 
        type=str,
        default='./result/result_bs1', 
        help='path to inference result.'
        )

    args = parser.parse_args()

    return args

