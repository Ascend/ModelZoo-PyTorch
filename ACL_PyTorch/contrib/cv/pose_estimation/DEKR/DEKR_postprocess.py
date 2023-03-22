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

import sys
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from tool import get_multi_scale_size, resize_align_multi_scale
sys.path.append(r"./DEKR")
from tools import _init_paths
from lib.config import update_config, cfg
from lib.dataset.transforms import FLIP_CONFIG
from lib.dataset import make_test_dataloader
from lib.utils.utils import create_logger
from lib.core.inference import offset_to_pose, aggregate_results
from lib.core.nms import pose_nms
from lib.core.match import match_pose_to_heatmap
from lib.utils.transforms import get_final_preds
from lib.utils.rescore import rescore_valid


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--flip-dir', type=str, required=True)
    parser.add_argument('--unflip-dir', type=str, required=True)
    parser.add_argument('--bs', type=int, default=1)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def get_output_data(dump_dir, idx):
    heatmap_file = os.path.join(dump_dir, "{:0>12d}_0.npy".format(idx))
    offset_file = os.path.join(dump_dir, "{:0>12d}_1.npy".format(idx))
    heatmap_data = np.load(heatmap_file)
    offset_data = np.load(offset_file)

    heatmap_data = torch.tensor(heatmap_data, dtype=torch.float32)
    offset_data = torch.tensor(offset_data, dtype=torch.float32)

    return heatmap_data, offset_data


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


def postprocess(config, final_output_dir):
    scale_list = (512, 768, 1024)
    data_loader, test_dataset = make_test_dataloader(config)

    all_reg_preds = []
    all_reg_scores = []

    pbar = tqdm(total=len(test_dataset), desc='Postprocessing')
    for idx, images in enumerate(data_loader):
        image = images[0].cpu().numpy()
        base_size, center, scale = get_multi_scale_size(
            image, config.DATASET.INPUT_SIZE, 1.0, 1.0, scale_list
        )
        heatmap_sum = 0
        poses = []

        image_resized, center, scale_resized = resize_align_multi_scale(
            image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0, scale_list
        )

        heatmap, offset = get_output_data(opt.dump_dir, idx)
        posemap = offset_to_pose(offset, flip=False)

        flip_index_heat = FLIP_CONFIG['COCO_WITH_CENTER']
        flip_index_offset = FLIP_CONFIG['COCO']

        heatmap_flip, offset_flip = get_output_data(opt.dump_dir_flip, idx)
        heatmap_flip = torch.flip(heatmap_flip, [3])
        heatmap = (heatmap + heatmap_flip[:, flip_index_heat, :, :]) / 2.0

        posemap_flip = offset_to_pose(offset_flip, flip_index=flip_index_offset)
        posemap = (posemap + torch.flip(posemap_flip, [3])) / 2.0

        heatmap_sum, poses = aggregate_results(
            cfg, heatmap_sum, poses, heatmap, posemap, 1.0
        )

        heatmap_avg = heatmap_sum / len(cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(cfg, heatmap_avg, poses)

        if len(scores) == 0:
            all_reg_preds.append([])
            all_reg_scores.append([])
        else:
            if cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

            final_poses = get_final_preds(
                poses, center, scale_resized, base_size
            )
            if cfg.RESCORE.VALID:
                scores = rescore_valid(cfg, final_poses, scores)
            all_reg_preds.append(final_poses)
            all_reg_scores.append(scores)

        pbar.update()

    sv_all_preds = [all_reg_preds]
    sv_all_scores = [all_reg_scores]
    sv_all_name = [cfg.NAME]

    pbar.close()

    for i in range(len(sv_all_preds)):
        print('Testing '+sv_all_name[i])
        preds = sv_all_preds[i]
        scores = sv_all_scores[i]
        if cfg.RESCORE.GET_DATA:
            test_dataset.evaluate(
                cfg, preds, scores, final_output_dir, sv_all_name[i]
            )
            print('Generating dataset for rescorenet successfully')
        else:
            name_values, _ = test_dataset.evaluate(
                cfg, preds, scores, final_output_dir, sv_all_name[i]
            )

            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(logger, name_value, cfg.MODEL.NAME)
            else:
                _print_name_value(logger, name_values, cfg.MODEL.NAME)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=
                        "./DEKR/experiments/coco/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_coco_x140.yaml",
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--dump_dir', dest='dump_dir',
                        help='dump dir for bin files',
                        required=True,
                        type=str)
    parser.add_argument('--dump_dir_flip', dest='dump_dir_flip',
                        help='dump dir for fliped bin files',
                        required=True,
                        type=str)

    opt = parser.parse_args()

    update_config(cfg, opt)
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, opt.cfg, 'valid'
    )

    final_output_dir = os.path.join('DEKR', final_output_dir)
    postprocess(cfg, final_output_dir)
