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

import os
import sys
sys.path.append('./exps/mspn.2xstg.coco/')
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import json
import time

import torch
import torch.distributed as dist

from cvpack.utils.logger import get_logger

from config import cfg
from network import MSPN
from lib.utils.dataloader import get_test_loader
from lib.utils.comm import is_main_process, synchronize, all_gather
from lib.utils.transforms import flip_back
from dataset.COCO.coco import COCODataset

def get_results(outputs, centers, scales, kernel=11, shifts=[0.25]):
    scales *= 200
    nr_img = outputs.shape[0]
    preds = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 2))
    maxvals = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 1))
    for i in range(nr_img):
        score_map = outputs[i].copy()
        score_map = score_map / 255 + 0.5
        kps = np.zeros((cfg.DATASET.KEYPOINT.NUM, 2))
        scores = np.zeros((cfg.DATASET.KEYPOINT.NUM, 1))
        border = 10
        dr = np.zeros((cfg.DATASET.KEYPOINT.NUM,
            cfg.OUTPUT_SHAPE[0] + 2 * border, cfg.OUTPUT_SHAPE[1] + 2 * border))
        dr[:, border: -border, border: -border] = outputs[i].copy()
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            dr[w] = cv2.GaussianBlur(dr[w], (kernel, kernel), 0)
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            for j in range(len(shifts)):
                if j == 0:
                    lb = dr[w].argmax()
                    y, x = np.unravel_index(lb, dr[w].shape)
                    dr[w, y, x] = 0
                    x -= border
                    y -= border
                lb = dr[w].argmax()
                py, px = np.unravel_index(lb, dr[w].shape)
                dr[w, py, px] = 0
                px -= border + x
                py -= border + y
                ln = (px ** 2 + py ** 2) ** 0.5
                if ln > 1e-3:
                    x += shifts[j] * px / ln
                    y += shifts[j] * py / ln
            x = max(0, min(x, cfg.OUTPUT_SHAPE[1] - 1))
            y = max(0, min(y, cfg.OUTPUT_SHAPE[0] - 1))
            kps[w] = np.array([x * 4 + 2, y * 4 + 2])
            scores[w, 0] = score_map[w, int(round(y) + 1e-9), \
                    int(round(x) + 1e-9)]
        # aligned or not ...
        kps[:, 0] = kps[:, 0] / cfg.INPUT_SHAPE[1] * scales[i][0] + \
                centers[i][0] - scales[i][0] * 0.5
        kps[:, 1] = kps[:, 1] / cfg.INPUT_SHAPE[0] * scales[i][1] + \
                centers[i][1] - scales[i][1] * 0.5
        preds[i] = kps
        maxvals[i] = scores 
    
    return preds, maxvals


def compute_on_dataset(data_loader, device="cpu"):
    results = list() 
    cpu_device = torch.device("cpu")

    results = list() 
    data = tqdm(data_loader) if is_main_process() else data_loader
    k = 0
    for _, batch in enumerate(data):
        imgs, scores, centers, scales, img_ids = batch
        output_name='img_%d_%d_1.bin' %(int(img_ids[0]), k)
        output_path=os.path.join('result/dumpOutput_device0/',output_name)
        outputs = np.fromfile(output_path, dtype=np.float32).reshape(1,17,64,48)
        k += 1

        centers = np.array(centers)
        scales = np.array(scales)
        preds, maxvals = get_results(outputs, centers, scales,
                cfg.TEST.GAUSSIAN_KERNEL, cfg.TEST.SHIFT_RATIOS)
        
        kp_scores = maxvals.squeeze(-1).mean(axis=1)
        preds = np.concatenate((preds, maxvals), axis=2)

        for i in range(preds.shape[0]):
            keypoints = preds[i].reshape(-1).tolist()
            score = scores[i] * kp_scores[i]
            image_id = img_ids[i]

            results.append(dict(image_id=image_id,
                                category_id=1,
                                keypoints=keypoints,
                                score=score))
    return results


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, logger):
    if is_main_process():
        logger.info("Accumulating ...")
    all_predictions = all_gather(predictions_per_gpu)

    if not is_main_process():
        return

    predictions = list()
    for p in all_predictions:
        predictions.extend(p)
    
    return predictions


def inference(data_loader, logger, device="cpu"):
    predictions = compute_on_dataset(data_loader, device)
    synchronize()
    predictions = _accumulate_predictions_from_multiple_gpus(
            predictions, logger)

    if not is_main_process():
        return

    return predictions    
     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--iter", "-i", type=int, default=-1)
    parser.add_argument("--datasets_path",default="$MSPN_HOME/dataset/COCO")
    args = parser.parse_args()
    COCODataset.cur_dir=os.path.join(args.datasets_path)
    num_gpus = int(
            os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed =  num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    if is_main_process() and not os.path.exists(cfg.TEST_DIR):
        os.mkdir(cfg.TEST_DIR)
    logger = get_logger(
            cfg.DATASET.NAME, cfg.TEST_DIR, args.local_rank, 'test_log.txt')

    if args.iter == -1:
        logger.info("Please designate one iteration.")

    data_loader = get_test_loader(cfg, num_gpus, args.local_rank, 'val',
            is_dist=distributed)

    device = 'cpu'
    results = inference(data_loader, logger, device)
    synchronize()

    if is_main_process():
        logger.info("Dumping results ...")
        results.sort(
                key=lambda res:(res['image_id'], res['score']), reverse=True) 
        results_path = os.path.join(cfg.TEST_DIR, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        logger.info("Get all results.")

        data_loader.ori_dataset.evaluate(results_path)


if __name__ == '__main__':
    begin = time.time()
    main()
    end = time.time()
    print('postprocess finished in', str(end - begin), 'seconds')