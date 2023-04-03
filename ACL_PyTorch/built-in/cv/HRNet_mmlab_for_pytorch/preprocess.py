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
import argparse
import copy
import json
import os
import stat
import warnings

import numpy as np
import torch

from mmcv import Config, DictAction
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
    parser.add_argument('--pre_data', help='the dir to save preprocess results')
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


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.data.train.pipeline
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=False),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # save the data
    out_path = args.pre_data
    data_path1 = os.path.join(out_path, 'data1')
    data_path2 = os.path.join(out_path, 'data2')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        os.mkdir(data_path1)
        os.mkdir(data_path2)
    i = 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(os.path.join(out_path, 'label.json'), flags, modes), 'w') as file_:
        for data in data_loader:
            with torch.no_grad():
                img_metas = data['img_metas'].data[0][0]
                aug_data = img_metas['aug_data']
                base_size = img_metas['base_size']
                center = img_metas['center']
                scale = img_metas['scale']
                flip_index = img_metas['flip_index']
                image_paths = img_metas['image_file']

                save_path1 = os.path.join(data_path1, '{}_{}'.format(base_size[1], base_size[0]))
                save_path2 = os.path.join(data_path2, '{}_{}'.format(base_size[1], base_size[0]))
                if not os.path.exists(save_path1):
                    os.mkdir(save_path1)
                if not os.path.exists(save_path2):
                    os.mkdir(save_path2)

                img_name = image_paths.split('/')[-1].split('.')[0]
                print('now is processing {}:{}'.format(i, img_name))
                i += 1
                img_dict = {'img_name': img_name, 'base_size': base_size, 'center': center.tolist(),
                            'scale': scale.tolist(), 'flip_index': flip_index, 'image_paths': image_paths}
                json_str = json.dumps(img_dict)
                file_.write(json_str)
                file_.write('\n')

                np.save(os.path.join(save_path1, f'{img_name}.npy'), aug_data[0].cpu().numpy())

                another_data = torch.flip(aug_data[0], [3])
                np.save(os.path.join(save_path2, f'{img_name}.npy'), another_data.cpu().numpy())
    print('success preprocess')


if __name__ == '__main__':
    main()
