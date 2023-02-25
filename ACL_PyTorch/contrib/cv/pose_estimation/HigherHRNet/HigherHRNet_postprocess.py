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
import argparse
import sys

from tqdm import tqdm
import numpy as np
import torch
sys.path.append('./HigherHRNet-Human-Pose-Estimation')
from lib.utils.utils import create_logger
from lib.utils.transforms import get_final_preds
from lib.dataset.transforms.build import FLIP_CONFIG
from lib.config import cfg
from lib.core.group import HeatmapParser
from lib.config import update_config
from lib.dataset.build import make_test_dataloader
from lib.core.inference import aggregate_results
from lib.utils.transforms import get_multi_scale_size


def get_output_data(dump_dir, idx):
    input_data = []
    input_1_file = os.path.join(dump_dir, "{:0>12d}_0.npy".format(idx))
    input_2_file = os.path.join(dump_dir, "{:0>12d}_1.npy".format(idx))
    input_data_1 = np.load(input_1_file)
    input_data_2 = np.load(input_2_file)

    input_data_1 = torch.tensor(input_data_1, dtype=torch.float32)
    input_data_2 = torch.tensor(input_data_2, dtype=torch.float32)

    input_data.append(input_data_1)
    input_data.append(input_data_2)
    return input_data


def _print_name_value(log, results, full_arch_name):
    names = results.keys()
    values = results.values()
    num_values = len(results)
    log.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    log.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    log.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


def postprocess(config, output_dir):
    scale_list = (512, 576, 640, 704, 768, 832, 896, 960, 1024)
    data_loader, test_dataset = make_test_dataloader(config)
    heatmapparser = HeatmapParser(config)
    all_preds = []
    all_scores = []
    for idx, (images, annos) in enumerate(tqdm(data_loader)):
        image = images[0].cpu().numpy()
        base_size, center, scale = get_multi_scale_size(
            image, config.DATASET.INPUT_SIZE, 1.0, min(
                config.TEST.SCALE_FACTOR), scale_list
        )
        final_heatmaps = None
        tags_list = []

        heatmaps_avg = 0
        num_heatmaps = 0
        heatmaps = []
        tags = []
        outputs = get_output_data(opt.dump_dir, idx)
        for i, output in enumerate(outputs):
            if len(outputs) > 1 and i != len(outputs) - 1:
                output = torch.nn.functional.interpolate(
                    output,
                    size=(outputs[-1].size(2), outputs[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                )

            offset_feat = config.DATASET.NUM_JOINTS \
                if config.LOSS.WITH_HEATMAPS_LOSS[i] else 0

            if config.LOSS.WITH_HEATMAPS_LOSS[i] and config.TEST.WITH_HEATMAPS[i]:
                heatmaps_avg += output[:, :config.DATASET.NUM_JOINTS]
                num_heatmaps += 1

            if config.LOSS.WITH_AE_LOSS[i] and config.TEST.WITH_AE[i]:
                tags.append(output[:, offset_feat:])
        if num_heatmaps > 0:
            heatmaps.append(heatmaps_avg / num_heatmaps)

        dataset_name = 'COCO'
        flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER'] \
            if config.DATASET.WITH_CENTER else FLIP_CONFIG[dataset_name]

        heatmaps_avg = 0
        num_heatmaps = 0
        outputs_flip = get_output_data(opt.dump_dir_flip, idx)

        for i in range(len(outputs_flip)):
            output = outputs_flip[i]
            if len(outputs_flip) > 1 and i != len(outputs_flip) - 1:
                output = torch.nn.functional.interpolate(
                    output,
                    size=(outputs_flip[-1].size(2), outputs_flip[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            output = torch.flip(output, [3])
            outputs.append(output)

            offset_feat = config.DATASET.NUM_JOINTS \
                if config.LOSS.WITH_HEATMAPS_LOSS[i] else 0

            if config.LOSS.WITH_HEATMAPS_LOSS[i] and config.TEST.WITH_HEATMAPS[i]:
                heatmaps_avg += \
                    output[:, :config.DATASET.NUM_JOINTS][:, flip_index, :, :]
                num_heatmaps += 1

            if config.LOSS.WITH_AE_LOSS[i] and config.TEST.WITH_AE[i]:
                tags.append(output[:, offset_feat:])
                if config.MODEL.TAG_PER_JOINT:
                    tags[-1] = tags[-1][:, flip_index, :, :]

        heatmaps.append(heatmaps_avg / num_heatmaps)

        if config.DATASET.WITH_CENTER and config.TEST.IGNORE_CENTER:
            heatmaps = [hms[:, :-1] for hms in heatmaps]
            tags = [tms[:, :-1] for tms in tags]
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(base_size[1], base_size[0]),
                mode='bilinear',
                align_corners=False
            )
            for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(base_size[1], base_size[0]),
                mode='bilinear',
                align_corners=False
            )
            for tms in tags
        ]
        final_heatmaps, tags_list = aggregate_results(
            config, 1.0, final_heatmaps, tags_list, heatmaps, tags
        )
        tags = torch.cat(tags_list, dim=4)
        grouped, scores = heatmapparser.parse(
            final_heatmaps, tags, config.TEST.ADJUST, config.TEST.REFINE
        )
        final_results = get_final_preds(
            grouped, center, scale, [
                final_heatmaps.size(3), final_heatmaps.size(2)]
        )

        all_preds.append(final_results)
        all_scores.append(scores)

    evaluate_results, _ = test_dataset.evaluate(
        config, all_preds, all_scores, output_dir
    )
    return evaluate_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="HigherHRNet-Human-Pose-Estimation/experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml",
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
    name_values = postprocess(cfg, final_output_dir)
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(logger, name_value, cfg.MODEL.NAME)
    else:
        _print_name_value(logger, name_values, cfg.MODEL.NAME)
