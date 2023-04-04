# Copyright 2023 Huawei Technologies Co., Ltd
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

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
import threading
import warnings

import mmcv
import numpy as np
import torch

from mmcv import Config, DictAction
from mmpose.core.evaluation import (aggregate_scale, aggregate_stage_flip,
                                    flip_feature_maps, get_group_preds,
                                    split_ae_outputs)
from mmpose.core.post_processing.group import HeatmapParser
from mmpose.datasets import build_dataloader, build_dataset


try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument('--label_dir', help='tmp dir for label')
    parser.add_argument('--dataset', help='tmp dir for dataset')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def postprocess(info, data_path, test_cfg):
    data_path1 = os.path.join(data_path, 'res1', info['img_name']) + '_0.npy'
    aug_data = np.load(data_path1)
    outputs = [aug_data]
    outputs[0] = torch.from_numpy(outputs[0])

    test_scale_factor = [1]

    base_size = info['base_size']
    center = np.array(info['center'])
    scale =  np.array(info['scale'])

    result = {}

    scale_heatmaps_list = []
    scale_tags_list = []
    parser = HeatmapParser(test_cfg)

    for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):

        heatmaps, tags = split_ae_outputs(
            outputs, test_cfg['num_joints'],
            test_cfg['with_heatmaps'], test_cfg['with_ae'],
            test_cfg.get('select_output_index', range(len(outputs))))

        if test_cfg.get('flip_test', True):
            # use flip test
            data_path2 = os.path.join(data_path, 'res2', info['img_name']) + '_0.npy'
            outputs_flipped = [np.load(data_path2)]
            outputs_flipped[0] = torch.from_numpy(outputs_flipped[0])

            heatmaps_flipped, tags_flipped = split_ae_outputs(
                outputs_flipped, test_cfg['num_joints'],
                test_cfg['with_heatmaps'], test_cfg['with_ae'],
                test_cfg.get('select_output_index', range(len(outputs))))

            heatmaps_flipped = flip_feature_maps(
                heatmaps_flipped, flip_index=info['flip_index'])
            if test_cfg['tag_per_joint']:
                tags_flipped = flip_feature_maps(
                    tags_flipped, flip_index=info['flip_index'])
            else:
                tags_flipped = flip_feature_maps(
                    tags_flipped, flip_index=None, flip_output=True)

        else:
            heatmaps_flipped = None
            tags_flipped = None

        aggregated_heatmaps = aggregate_stage_flip(
            heatmaps,
            heatmaps_flipped,
            index=-1,
            project2image=test_cfg['project2image'],
            size_projected=base_size,
            align_corners=test_cfg.get('align_corners', True),
            aggregate_stage='average',
            aggregate_flip='average')

        aggregated_tags = aggregate_stage_flip(
            tags,
            tags_flipped,
            index=-1,
            project2image=test_cfg['project2image'],
            size_projected=base_size,
            align_corners=test_cfg.get('align_corners', True),
            aggregate_stage='concat',
            aggregate_flip='concat')

        if s == 1 or len(test_scale_factor) == 1:
            if isinstance(aggregated_tags, list):
                scale_tags_list.extend(aggregated_tags)
            else:
                scale_tags_list.append(aggregated_tags)

        if isinstance(aggregated_heatmaps, list):
            scale_heatmaps_list.extend(aggregated_heatmaps)
        else:
            scale_heatmaps_list.append(aggregated_heatmaps)

    aggregated_heatmaps = aggregate_scale(
        scale_heatmaps_list,
        align_corners=test_cfg.get('align_corners', True),
        aggregate_scale='average')

    aggregated_tags = aggregate_scale(
        scale_tags_list,
        align_corners=test_cfg.get('align_corners', True),
        aggregate_scale='unsqueeze_concat')

    heatmap_size = aggregated_heatmaps.shape[2:4]
    tag_size = aggregated_tags.shape[2:4]
    if heatmap_size != tag_size:
        tmp = []
        for idx in range(aggregated_tags.shape[-1]):
            tmp.append(
                torch.nn.functional.interpolate(
                    aggregated_tags[..., idx],
                    size=heatmap_size,
                    mode='bilinear',
                    align_corners=test_cfg.get('align_corners', True)).unsqueeze(-1))
        aggregated_tags = torch.cat(tmp, dim=-1)

    # perform grouping
    grouped, scores = parser.parse(aggregated_heatmaps, aggregated_tags,
                                   test_cfg['adjust'],
                                   test_cfg['refine'])

    preds = get_group_preds(
        grouped,
        center,
        scale, [aggregated_heatmaps.size(3),
                aggregated_heatmaps.size(2)],
        use_udp=False)

    image_paths = []
    image_paths.append(info['image_paths'])
    return_heatmap = False
    if return_heatmap:
        output_heatmap = aggregated_heatmaps.detach().cpu().numpy()
    else:
        output_heatmap = None

    result['preds'] = preds
    result['scores'] = scores
    result['image_paths'] = image_paths
    result['output_heatmap'] = output_heatmap

    return result


def read_info(json_path):
    res = []
    if not os.path.exists(json_path):
        print(json_path, 'is not exist')

    with open(json_path, 'r') as f:
        for json_obj in f:
            temp = json.loads(json_obj)
            res.append(temp)
    return res


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    file_info = read_info(args.label_dir)
    outputs = []

    def thread_test(info, data, test_cfg):
        res = postprocess(info, data, test_cfg)
        outputs.append(res)

    threads = []
    index = 0
    for i in file_info:
        print('postprocessing index:', index)
        thread = threading.Thread(target=thread_test(i, args.dataset, cfg.model['test_cfg']))
        thread.start()
        threads.append(thread)
        index += 1

    for thread in threads:
        thread.join()

    rank = 0
    eval_config = cfg.get('evaluation', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    if rank == 0:

        results = dataset.evaluate(outputs, cfg.work_dir, **eval_config)
        for k, v in sorted(results.items()):
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
