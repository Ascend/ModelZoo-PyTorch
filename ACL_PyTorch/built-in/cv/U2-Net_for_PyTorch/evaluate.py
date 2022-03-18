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


import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import os
import json
import argparse

eps = 2.2204e-16


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluation for U-2-Net'
    )
    parser.add_argument('--res_dir', type=str, default='./test_vis_ECSSD',
                        help='inference result image dir')
    parser.add_argument('--gt_dir', type=str, default='./datasets/ECSSD/ground_truth_mask',
                        help='gt dir for dataset')
    args = parser.parse_args()
    return args


def parameter():
    p = {}
    p['gtThreshold'] = 0.5
    p['beta'] = np.sqrt(0.3)
    p['thNum'] = 100
    p['thList'] = np.linspace(0, 1, p['thNum'])

    return p


def im2double(im):
    return cv2.normalize(im.astype('float'),
                         None,
                         0.0, 1.0,
                         cv2.NORM_MINMAX)


def prCount(gtMask, curSMap, p):
    gtH, gtW = gtMask.shape[0:2]
    algH, algW = curSMap.shape[0:2]

    if gtH != algH or gtW != algW:
        curSMap = cv2.resize(curSMap, (gtW, gtH))

    gtMask = (gtMask >= p['gtThreshold']).astype(np.float32)
    gtInd = np.where(gtMask > 0)
    gtCnt = np.sum(gtMask)

    if gtCnt == 0:
        prec = []
        recall = []
    else:
        hitCnt = np.zeros((p['thNum'], 1), np.float32)
        algCnt = np.zeros((p['thNum'], 1), np.float32)

        for k, curTh in enumerate(p['thList']):
            thSMap = (curSMap >= curTh).astype(np.float32)
            hitCnt[k] = np.sum(thSMap[gtInd])
            algCnt[k] = np.sum(thSMap)

        prec = hitCnt / (algCnt+eps)
        recall = hitCnt / gtCnt

    return prec, recall


def PR_Curve(resDir, gtDir, method):
    p = parameter()
    beta = p['beta']

    gtImgs = glob(gtDir + '/*.png')

    prec = []
    recall = []

    for gtName in tqdm(gtImgs):
        dir, name = os.path.split(gtName)
        mapName = os.path.join(resDir, name)

        curMap = im2double(cv2.imread(mapName, cv2.IMREAD_GRAYSCALE))

        curGT = im2double(cv2.imread(gtName, cv2.IMREAD_GRAYSCALE))

        if curMap.shape[0] != curGT.shape[0]:
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[2]))

        curPrec, curRecall = prCount(curGT, curMap, p)

        prec.append(curPrec)
        recall.append(curRecall)


    prec = np.hstack(prec[:])
    recall = np.hstack(recall[:])

    prec = np.mean(prec, 1)
    recall = np.mean(recall, 1)

    # compute the max F-Score
    score = (1+beta**2)*prec*recall / (beta**2*prec + recall)
    curTh = np.argmax(score)
    curScore = np.max(score)

    res = {}
    res['prec'] = prec
    res['recall'] = recall
    res['curScore'] = curScore
    res['curTh'] = curTh

    return res


def MAE_Value(resDir, gtDir, method):
    p = parameter()
    gtThreshold = p['gtThreshold']

    gtImgs = glob(gtDir + '/*.png')

    MAE = []

    for gtName in gtImgs:
        dir, name = os.path.split(gtName)
        mapName= os.path.join(resDir, name)

        # mapName = mapName.replace('.png', '_' + method + '.png')
        curMap = im2double(cv2.imread(mapName, cv2.IMREAD_GRAYSCALE))

        curGT = im2double(cv2.imread(gtName, cv2.IMREAD_GRAYSCALE))
        curGT = (curGT >= gtThreshold).astype(np.float32)

        if curMap.shape[0] != curGT.shape[0]:
            curMap = cv2.resize(curMap, (curGT.shape[1], curGT.shape[2]))

        diff = np.abs(curMap - curGT)

        MAE.append(np.mean(diff))

    return np.mean(MAE)


if __name__ == "__main__":
    args = parse_args()
    res_dir = args.res_dir
    gt_dir = args.gt_dir
    mae = MAE_Value(res_dir, gt_dir, '')
    pr = PR_Curve(res_dir, gt_dir, '')
    result = {
        'max F': float(pr['curScore']),
        'MAE': float(mae)
    }
    print("evaluate result:", result)
    with open('result.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
