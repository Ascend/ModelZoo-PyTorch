# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import time

#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from glow import WaveGlow, WaveGlowLoss
from mel2samp import Mel2Samp

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading)
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WaveGlow(**waveglow_config).to("npu:0")
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving.state_dict(),
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def train(num_gpus, rank, group_name, step_stop, output_directory, epochs, learning_rate,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard):
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus == 1:
        LOCAL_RANK = 0
    elif num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
        LOCAL_RANK = torch.distributed.get_rank()
    #=====END:   ADDED FOR DISTRIBUTED======

    criterion = WaveGlowLoss(sigma)
    model = WaveGlow(**waveglow_config).to("npu:{}".format(LOCAL_RANK))

    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======

    from apex.optimizers import NpuFusedAdam
    optimizer = NpuFusedAdam(model.parameters(),lr=learning_rate)

    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2',loss_scale = 128.0)
    
    # Add DDP
    if(num_gpus > 1):
        model = DistributedDataParallel(model,
                                        device_ids=[LOCAL_RANK], 
                                        broadcast_buffers=False)

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                                      optimizer)
        iteration += 1  # next iteration is iteration + 1

    trainset = Mel2Samp(**data_config)
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory, 'logs'))

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        if(num_gpus > 1):
            train_sampler.set_epoch(epoch)
        iter_start_time = time.time()
        fps_start_step = 0

        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            if(fps_start_step == 5):
                iter5_start_time = time.time()
            model.zero_grad()

            mel, audio = batch
            mel = torch.autograd.Variable(mel.to("npu:{}".format(LOCAL_RANK)))
            audio = torch.autograd.Variable(audio.to("npu:{}".format(LOCAL_RANK)))
            
            outputs = model((mel, audio))

            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            # Step time
            print("{}:\tloss:{:.9f}\ttime:{:.9f}\t".format(iteration, 
                                                           reduced_loss, 
                                                           time.time()-iter_start_time))
            iter_start_time = time.time()

            if with_tensorboard and rank == 0:
                logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)

            if (iteration % iters_per_checkpoint == 0):
                if rank == 0:
                    checkpoint_path = "{}/waveglow_{}".format(
                        output_directory, iteration)
                    if(num_gpus == 1):
                        save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)
                    else:
                        save_checkpoint(model.module, optimizer, learning_rate, iteration,
                                    checkpoint_path)
            iteration += 1
            fps_start_step += 1
            #=============== Step Stop for CANNProfiling ==============
            if step_stop == 1:
                break
        if step_stop == 1:
            break
        # FPS 
        print("Epoch {} FPS:{:.9f}".format(epoch, (batch_size * num_gpus)/(time.time() - iter5_start_time)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('-step_stop', type=int,default=0,
                        help='step stop for CANNProfiling')
    parser.add_argument('-fp16_run', type=bool, default=True,
                        help='for apex mixed training')
    parser.add_argument('-output_directory', type=str, default='checkpoints',
                        help='output directory of checkpoints pth')
    parser.add_argument('-epochs', type=int, default=100000,
                        help='training epochs')
    parser.add_argument('-learning_rate', type=float, default=1e-4,
                        help='learing rate')
    parser.add_argument('-sigma', type=float, default=1.0,
                        help='sigma value')
    parser.add_argument('-iters_per_checkpoint', type=int, default=2000,
                        help='iterations of checkpoints')
    parser.add_argument('-batch_size', type=int, default=12,
                        help='batch size')
    parser.add_argument('-seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('-checkpoint_path', type=str, default='',
                        help='path of pre-trained models')
    parser.add_argument('-with_tensorboard', action="store_true")
    
    args = parser.parse_args()
    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    # train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global waveglow_config
    waveglow_config = config["waveglow_config"]

    num_gpus = torch.npu.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple NPUs detected but no distributed group set")
            print("Only running 1 NPU.  Use distributed.py for multiple NPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single NPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(num_gpus, args.rank, args.group_name, args.step_stop, 
          args.output_directory, args.epochs, args.learning_rate,
          args.sigma, args.iters_per_checkpoint, args.batch_size, 
          args.seed, args.fp16_run, args.checkpoint_path, args.with_tensorboard)
