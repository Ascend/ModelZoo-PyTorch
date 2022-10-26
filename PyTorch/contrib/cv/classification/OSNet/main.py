"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the BSD 3-Clause License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://spdx.org/licenses/BSD-3-Clause.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys
import time
import os.path as osp
import argparse
import torch
if torch.__version__>= '1.8':
      import torch_npu
import torch.nn as nn
import os

import torchreid
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs, evaluate_kwargs
)

from apex import amp


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                use_npu=cfg.use_npu,
                label_smooth=cfg.loss.softmax.label_smooth,
                use_amp=cfg.amp
            )

        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--device_num', default=-1, type=int,
                        help='device_num')
    parser.add_argument('--gpu', action='store_true',
                        help="gpu")
    parser.add_argument('--npu', action='store_true',
                        help="npu")
    parser.add_argument('--amp', action='store_true',
                        help="amp")
    parser.add_argument('--addr', default='127.0.0.1',
                    type=str, help='master addr')
    parser.add_argument('--ignore_classifer', action='store_true',
                    help="ignore classifer layer weight when loading pretrained model")
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = args.gpu
    cfg.use_npu = args.npu
    cfg.device_num = args.device_num
    cfg.local_rank = args.local_rank
    cfg.amp = args.amp
    cfg.addr = args.addr
    cfg.ignore_classifer = args.ignore_classifer
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.device_num == -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
        os.environ['device_num'] = '-1'
    else:
        environ_str = '0'
        for i in range(1, cfg.device_num):
            environ_str = environ_str + ',%d' % i
        os.environ["CUDA_VISIBLE_DEVICES"] = environ_str
        os.environ['device_num'] = str(cfg.device_num)
    
    if cfg.use_gpu:
        if cfg.device_num > 1:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.manual_seed_all(cfg.train.seed)
        torch.cuda.set_device(cfg.local_rank)
        os.environ['device'] = 'gpu'
        torch.backends.cudnn.benchmark = True

    if cfg.use_npu:
        os.environ['MASTER_ADDR'] = cfg.addr
        os.environ['MASTER_PORT'] = '29688'
        if cfg.device_num > 1:
            torch.distributed.init_process_group(backend='hccl', rank=args.local_rank, world_size=args.device_num)
        torch.npu.manual_seed_all(cfg.train.seed)
        torch.npu.set_device(cfg.local_rank)
        os.environ['device'] = 'npu'

    os.environ['batch_size'] = str(cfg.train.batch_size) 
    datamanager = build_datamanager(cfg)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))


    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights, cfg.ignore_classifer)

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))

    if cfg.use_gpu:
        model = model.cuda()
    elif cfg.use_npu:
        model = model.npu()

    if cfg.amp:
        if cfg.use_npu:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic", combine_grad=True)
        elif cfg.use_gpu:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128.0)

    if cfg.device_num > 1:
        model = nn.parallel.DistributedDataParallel(model, 
                    device_ids=[cfg.local_rank], 
                    output_device=cfg.local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=False
                    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    if cfg.test.evaluate:
        model.eval()
        engine.test(**evaluate_kwargs(cfg))
        return
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
