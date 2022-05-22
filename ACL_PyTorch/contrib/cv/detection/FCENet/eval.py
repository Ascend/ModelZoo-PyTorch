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
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test

from mmocr.apis.test import single_gpu_test
from mmocr.apis.utils import (disable_text_recog_aug_test,
                              replace_image_to_tensor)
from mmocr.datasets import build_dataloader, build_dataset
from mmocr.models import build_detector
from mmocr.utils import revert_sync_batchnorm, setup_multi_processes

from tools.test import parse_args

def main():
    args = parse_args()

    assert (
        args.out or args.eval or args.format_only or args.show
        or args.show_dir), (
            'Please specify at least one operation (save/eval/format/show the '
            'results / save the results) with the argument "--out", "--eval"'
            ', "--format-only", "--show" or "--show-dir".')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified.')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.model.get('pretrained'):
        cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = (cfg.data.get('test_dataloader', {})).get(
        'samples_per_gpu', cfg.data.get('samples_per_gpu', 1))
    if samples_per_gpu > 1:
        cfg = disable_text_recog_aug_test(cfg)
        cfg = replace_image_to_tensor(cfg)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'workers_per_gpu',
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **cfg.data.get('test_dataloader', {}),
        **dict(samples_per_gpu=samples_per_gpu)
    }

    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = revert_sync_batchnorm(model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        is_kie = cfg.model.type in ['SDMGR']
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  is_kie, args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            output_file_path = './boundary_results.txt'
            boundary_results = []
            with open(output_file_path, 'r') as ff:
                temp = ff.readlines()
                for item in temp: 
                    tmp = eval(item)
                    boundary_results.append(tmp)
            
            print(dataset.evaluate(boundary_results, **eval_kwargs))


if __name__ == '__main__':
    main()