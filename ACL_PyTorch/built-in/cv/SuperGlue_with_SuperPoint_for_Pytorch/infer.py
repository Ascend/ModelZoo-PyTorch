# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
import time
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
from ais_bench.infer.interface import InferSession


def parse_args():
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--superpoint_path', type=str, default='superpoint.om',
        help='Path to the superpoint model')
    parser.add_argument(
        '--superglue_path', type=str, default='superglue.om',
        help='Path to the superglue model')
    parser.add_argument(
        '--device_id', type=int, default=0, help='Id of device')

    parser.add_argument(
        '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to preprocess')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1600, 1200],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')

    args = parser.parse_args()
    return args


def preprocess_image_pairs(opt, img_pairs):
    for idx, pair in enumerate(tqdm(img_pairs)):
        name0, name1 = pair[:2]
        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0
        # Load and preprocess the image pair.
        input_dir = Path(opt.input_dir)
        image0, inp0, _ = read_image(
            input_dir / name0, 'cpu', opt.resize, rot0, opt.resize_float)
        image1, inp1, _ = read_image(
            input_dir / name1, 'cpu', opt.resize, rot1, opt.resize_float)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        else:
            images0.append([inp0.numpy()])
            images1.append([inp1.numpy()])
    return images0, images1


def postprocess_match_results(opt, img_pairs, infer_results):
    
    pose_errors = []
    precisions = []
    matching_scores = []

    for idx, pair in tqdm(enumerate(img_pairs)):
        # Perform the matching.
        kpts0, kpts1 = features0[idx][0], features1[idx][0]
        matches, conf = infer_results[idx][0], infer_results[idx][2]
        matches = np.squeeze(matches, axis=0)

        # Keep the matching keypoints.
        valid = matches > -1 # 1 x num
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        
        name0, name1 = pair[:2]
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0
        input_dir = Path(opt.input_dir)
        image0, inp0, scales0 = read_image(
            input_dir / name0, 'cpu', opt.resize, rot0, opt.resize_float)
        image1, inp1, scales1 = read_image(
            input_dir / name1, 'cpu', opt.resize, rot1, opt.resize_float)

        # Estimate the pose and compute the pose error.
        assert len(pair) == 38, 'Pair does not have ground truth info'
        K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

        # Scale the intrinsics to resized image.
        K0 = scale_intrinsics(K0, scales0)
        K1 = scale_intrinsics(K1, scales1)

        # Update the intrinsics + extrinsics if EXIF rotation was found.
        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T_0to1
            if rot0 != 0:
                K0 = rotate_intrinsics(K0, image0.shape, rot0)
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                K1 = rotate_intrinsics(K1, image1.shape, rot1)
                cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            T_0to1 = cam1_T_cam0

        epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
        correct = epi_errs < 5e-4
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

        thresh = 1.  # In pixels relative to resized image size.
        ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        pose_error = np.maximum(err_t, err_R)
        pose_errors.append(pose_error)
        precisions.append(precision)
        matching_scores.append(matching_score)
        
    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    prec = 100.*np.mean(precisions)
    ms = 100.*np.mean(matching_scores)
    print('Evaluation Results (mean over {} pairs):'.format(len(img_pairs)))
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0], aucs[1], aucs[2], prec, ms))
       

def infer_dymShape(om_path, input_datas, outputSizes, device_id=0):
    # init session
    session = InferSession(device_id, om_path)

    # inference
    outputs_list = []
    for input_data in tqdm(input_datas):
        outputs = session.infer(input_data, mode='dymshape', custom_sizes=outputSizes)
        outputs_list.append(outputs)
    print('dymshape infer avg:{} ms'.format(np.mean(session.sumary().exec_time_list)))
    return outputs_list


if __name__=="__main__":
    
    option = parse_args()
    with open(option.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]
    if option.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), option.max_length])]
    if option.shuffle:
        random.Random(0).shuffle(pairs)

    start_time = time.time()

    print("Preprocess image pairs:")
    images0, images1 = [], []
    preprocess_image_pairs(option, pairs)
    
    print("Inference of SuperPoint:")
    features0 = infer_dymShape(
        om_path=option.superpoint_path,
        input_datas=images0,
        outputSizes=10000000,
        device_id=option.device_id,
        )
    features1 = infer_dymShape(
        om_path=option.superpoint_path,
        input_datas=images1,
        outputSizes=10000000,
        device_id=option.device_id,
        )
    
    print("Preprocess features:")
    features = [[*feat0, *feat1] for (feat0, feat1) in zip(features0, features1)]
    for feature in tqdm(features):
        for i, feat in enumerate(feature):
            feature[i] = np.expand_dims(feat, axis=0)

    print("Inference of SuperGlue:")
    results = infer_dymShape(
        om_path=option.superglue_path,
        input_datas=features,
        outputSizes=10000000,
        device_id=option.device_id,
        )
    
    print("Postprocess match results:")
    postprocess_match_results(option, pairs, results)

    e2e_time = time.time() - start_time
    print('E2E time(ms): {}s'.format(e2e_time * 1000 / len(images0)))
