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

import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import random
import numpy as np
import torch.distributed as dist

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    global model
    args.device = torch.device(f'npu:{args.rank_id}')
    torch.npu.set_device(args.device)

    args.is_master_node = not args.distributed or args.rank_id == 0
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        if args.is_master_node:
            print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            if args.distributed == 1:
                #print('init_process_group before')
                dist.init_process_group(backend="hccl", init_method='env://',
                                        world_size=args.n_GPUs, rank=args.rank_id)
                #print('init_process_group after')
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _model = _model.to(args.device)
            if args.is_master_node:
                print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters())/1000000.0))
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
