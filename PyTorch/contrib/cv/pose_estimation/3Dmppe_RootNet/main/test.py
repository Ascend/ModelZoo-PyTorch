# Copyright 2021 Huawei Technologies Co., Ltd
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
from tqdm import tqdm
import numpy as np
from config import cfg
import torch
from base import Tester
import torch.npu
import torch.backends.cudnn as cudnn
from utils.vis import vis_keypoints
import cv2
CALCULATE_DEVICE = "npu"


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--npu_device_test', default='0', type=str,
                        help='specifies the id of the NPU to use')
    parser.add_argument('--data_path', default='data', type=str,
                        help='location of the dataset')
    args = parser.parse_args()
    
    args.npu_device_test = CALCULATE_DEVICE + ':' + args.npu_device_test
    assert args.test_epoch, 'Test epoch is required.'

    return args


def main():

    args = parse_args()
    tester = Tester(args.test_epoch)
    tester.logger.info(args)
    cfg.set_args_test(args.data_path)

    cudnn.fastest = True
    cudnn.benchmark = True
    tester._make_batch_generator()
    tester._make_model(args.npu_device_test)

    preds = []
    with torch.no_grad():
        for itr, (input_img, cam_param) in enumerate(tqdm(tester.batch_generator)):
            
            input_img = input_img.to(args.npu_device_test, non_blocking=True)
            cam_param = cam_param.to(args.npu_device_test, non_blocking=True)
            coord_out = tester.model(input_img, cam_param).to(args.npu_device_test)
            coord_out = coord_out.cpu().numpy() 
            preds.append(coord_out)
            
    # evaluate
    preds = np.concatenate(preds, axis=0)
    tester._evaluate(preds, cfg.result_dir)    


if __name__ == "__main__":
    main()
