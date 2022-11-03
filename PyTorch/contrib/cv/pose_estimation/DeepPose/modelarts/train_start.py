# Copyright 2022 Huawei Technologies Co., Ltd
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
import copy
import os
import os.path as osp
import time
import glob
import os
import ast
import moxing as mox

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmpose.apis import train_model
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger

from selfexport import pth2onnx

__version__ = '0.13.0'


_CACHE_ROOT_URL = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir))
_CACHE_TRAIN_OUT_URL = os.path.join(_CACHE_ROOT_URL, 'output')
_CACHE_TRAIN_DATA_URL = os.path.join(_CACHE_ROOT_URL, 'data')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--train_url', type=str, default='',
                        help='the path model saved')
    parser.add_argument('--data_url', type=str, default='',
                        help='the training data')

    parser.add_argument('--work_dir', type=str, default = "output", help='the dir to save logs and models')
    parser.add_argument('--total_epochs', type=int, default = 1, help='epochs')
    parser.add_argument('--config', type=str, default ='../configs/top_down/deeppose/coco/npu_deeppose_res50_coco_256x192.py', 
                        help='train config file path')



    parser.add_argument('--onnx', default=True, type=ast.literal_eval,
                        help="convert pth model to onnx")
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--device',
        type=str,
        help='device'
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi','pytorch-npu'],
        default='none',
        help='job launcher')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def find_latest_pth_file(pth_save_dir):
    pth_pattern = os.path.join(pth_save_dir, '*.pth')
    pth_list = glob.glob(pth_pattern)
    if not pth_list:
        print(f"Cant't found pth in {pth_save_dir}")
        exit()
    pth_list.sort(key=os.path.getmtime)
    print("==================== %s will be exported to .onnx model next! ====================" % pth_list[-1])
    return os.path.join(pth_list[-1])


def main():
    args = parse_args()
    args.work_dir = _CACHE_TRAIN_OUT_URL
    os.makedirs(_CACHE_TRAIN_OUT_URL, mode=0o777, exist_ok=True)
    os.makedirs(_CACHE_TRAIN_DATA_URL, mode=0o777, exist_ok=True)
    mox.file.copy_parallel(args.data_url, _CACHE_TRAIN_DATA_URL)
    root_dir = _CACHE_ROOT_URL
    os.chdir(root_dir)
    cfg_script = os.path.join(root_dir, args.config)
    cfg = Config.fromfile(cfg_script)
    cfg.total_epochs = args.total_epochs
    cfg.data.train.ann_file = f'{_CACHE_TRAIN_DATA_URL}/annotations/person_keypoints_train2017.json'
    cfg.data.train.img_prefix = f'{_CACHE_TRAIN_DATA_URL}/train2017/'
    cfg.data.train.data_cfg.bbox_file = f'{_CACHE_TRAIN_DATA_URL}/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'
    cfg.data.val.ann_file = f'{_CACHE_TRAIN_DATA_URL}/annotations/person_keypoints_val2017.json'
    cfg.data.val.img_prefix = f'{_CACHE_TRAIN_DATA_URL}/val2017/'
    cfg.data.val.data_cfg.bbox_file = f'{_CACHE_TRAIN_DATA_URL}/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'
    cfg.data.test.ann_file = f'{_CACHE_TRAIN_DATA_URL}/annotations/person_keypoints_val2017.json'
    cfg.data.test.img_prefix = f'{_CACHE_TRAIN_DATA_URL}/val2017/'
    cfg.data.test.data_cfg.bbox_file = f'{_CACHE_TRAIN_DATA_URL}/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'
    
    print("**********************************8")
    print(cfg.data.train.data_cfg.bbox_file)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8
        
    ####    
    world_size=args.world_size
    ####
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher,world_size, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    '''
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    '''
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    '''
    logger.info(f'Config:\n{cfg.pretty_text}')
    '''
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic,gpu_npu=cfg.gpu_npu)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['rank'] = int(os.environ['LOCAL_RANK'])
    print("**********")
    print(os.environ['LOCAL_RANK'])
    meta['world_size'] = world_size
    meta['batch_size'] = cfg.data['samples_per_gpu']
    print("batch_size:",meta['batch_size'])
    
    model = build_posenet(cfg.model)
    
    datasets = [build_dataset(cfg.data.train)]
    if not distributed:
      if cfg.gpu_npu=='gpu':
          meta['dev']='gpu'
          cfg.device='cuda:0'
          model.cuda()
      else:
          meta['dev']='npu'
          cfg.device='npu:0'
          torch.npu.set_device(cfg.device)
          model.npu()
          print("use_gpu")
    else:
      if cfg.gpu_npu=='gpu':
          meta['dev']='gpu'
          model.cuda()
      else:
          meta['dev']='npu'
          model.npu()
          print("use_npu")
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmpose version, config file content
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmpose_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text,
        )
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

    try:
        if args.onnx:
            latest_pth_file = find_latest_pth_file(_CACHE_TRAIN_OUT_URL)
            pth2onnx(cfg_file=cfg_script,
                    pth_file=latest_pth_file,
                    output_file=os.path.join(_CACHE_TRAIN_OUT_URL, 'deeppose.onnx'),
                    input_shape=[32, 3, 256, 192])

    except Exception as exp:
        print(f"run {cmd} failed, {exp}")
        raise exp
    finally:
        mox.file.copy_parallel(_CACHE_TRAIN_OUT_URL, args.train_url)


if __name__ == '__main__':
    main()
