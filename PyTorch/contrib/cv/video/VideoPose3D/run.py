# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import errno
import math
import logging

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random, fetch, run_evaluation
from common.structure import AverageMeter, time_format_convert, device_id_to_process_device_map

from tensorboardX import SummaryWriter
from apex import amp
# from apex.optimizers import NpuFusedAdam


def main():
    args = parse_args()
    # print(args)

    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(f"args.output:{args.output}")	

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '27005'

    process_device_map = device_id_to_process_device_map(args.device_list)

    if args.device == 'npu':
        ngpus_per_node = len(process_device_map)
    else:
        ngpus_per_node = args.num_gpus
    
    args.num_gpus = ngpus_per_node
    args.world_size = args.world_size * ngpus_per_node

    # npu = int(os.environ['RANK_ID'])
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def setup_logger(final_output_dir, rank, phase):
    # time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_rank{}.log'.format(phase, rank)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    # logging.basicConfig(format=head)
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger
    

def main_worker(gpu, ngpus_per_node, args):
    process_device_map = device_id_to_process_device_map(args.device_list)
    log_dir = args.output
    logger = setup_logger(log_dir, gpu, 'train')

    # args.gpu = gpu
    args.gpu = process_device_map[gpu]
    # logger.info(f"args.gpu is {args.gpu}")

    args.rank = args.rank * ngpus_per_node + gpu

    # print(f'args.print_feq:{args.print_feq}')
    if args.rank % ngpus_per_node == 0:
        log_path = args.log
        writer_dict = {
            'writer': SummaryWriter(logdir=log_path),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

    if args.device == 'npu':
        print("args.rank:",args.rank)
        dist.init_process_group(backend=args.dist_backend, # init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    else:
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    logger.info(f'Loading dataset for rank:{args.rank}...')
    dataset_path = 'data/data_3d_' + args.dataset + '.npz'
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    elif args.dataset.startswith('humaneva'):
        from common.humaneva_dataset import HumanEvaDataset
        dataset = HumanEvaDataset(dataset_path)
    elif args.dataset.startswith('custom'):
        from common.custom_dataset import CustomDataset
        dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
    else:
        raise KeyError('Invalid dataset')

    logger.info(f'Preparing data for rank:{args.rank}...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    logger.info(f'Loading 2D detections for rank:{args.rank}...')
    keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue
                
            for cam_idx in range(len(keypoints[subject][action])):
                
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])
            
    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    subjects_train = args.subjects_train.split(',')
    subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
    if not args.render:
        subjects_test = args.subjects_test.split(',')
    else:
        subjects_test = [args.viz_subject]

    semi_supervised = len(subjects_semi) > 0
    if semi_supervised and not dataset.supports_semi_supervised():
        raise RuntimeError('Semi-supervised training is not implemented for this dataset')

    # moved fatch to utils.py

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)
        
    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, keypoints=keypoints, dataset=dataset, args=args ,action_filter=action_filter)

    filter_widths = [int(x) for x in args.architecture.split(',')]
    if not args.disable_optimizations and not args.dense and args.stride == 1:
        # Use optimized model for single-frame predictions
        model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)
    else:
        # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
        model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                    dense=args.dense)
        
    model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)

    receptive_field = model_pos.receptive_field()
    logger.info('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2 # Padding on each side
    if args.causal:
        logger.info('INFO: Using causal convolutions')
        causal_shift = pad
    else:
        causal_shift = 0

    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    assert args.gpu is not None, "Something wrong about args.gpu, it shouldn't be None."

    if not torch.npu.is_available():
        print("We only implemented for GPUs")
        raise NotImplementedError
    else:
        loc = f'npu:{args.gpu}'
        torch.npu.set_device(loc)
        model_pos = model_pos.to(loc)
        model_pos_train = model_pos_train.to(loc)
        model_pos = torch.nn.parallel.DistributedDataParallel(model_pos, device_ids=[args.gpu], broadcast_buffers=False)
        

        
    if args.evaluate:
        assert args.resume is ''
        chk_filename = os.path.join(args.checkpoint, args.evaluate)
        logger.info(f'Loading checkpoint {chk_filename}')
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_pos.load_state_dict(checkpoint['model_pos'])
        model_traj = None
            
        
    test_generator = UnchunkedGenerator(args, cameras_valid, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    logger.info('INFO: Testing on {} frames'.format(test_generator.num_frames()))

    if not args.evaluate:
        cameras_train, poses_train, poses_train_2d = fetch(subjects_train, keypoints=keypoints, dataset=dataset, args=args, action_filter=action_filter, subset=args.subset)

        lr = args.learning_rate
        if args.rank % args.num_gpus == 0:
            logger.info(f"inital learning rate is:{lr}")
        if semi_supervised:
            print("Not Implement semi_supervised version for DDP")
            raise NotImplementedError
        else:
            optimizer = optim.Adam(model_pos_train.parameters(), lr=lr) #, amsgrad=True)
            # optimizer = NpuFusedAdam(model_pos_train.parameters(), lr=lr)
        print(f"Use Apex:{args.apex}")
        print(f"Sampler:{args.sampler}")
        if args.apex:
            model_pos_train, optimizer = amp.initialize(model_pos_train, optimizer, opt_level="O1", loss_scale=128.0) #, combine_grad=True)
        model_pos_train = torch.nn.parallel.DistributedDataParallel(model_pos_train, device_ids=[args.gpu], broadcast_buffers=False)
            
        lr_decay = args.lr_decay

        losses_3d_train = []
        losses_3d_train_eval = []
        losses_3d_valid = []

        epoch = 0
        initial_momentum = 0.1
        final_momentum = 0.001
        
        
        train_generator = ChunkedGenerator(args, args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
                                        pad=pad, causal_shift=causal_shift, shuffle=True, random_seed=args.random_seed, augment=args.data_augmentation,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
        train_generator_eval = UnchunkedGenerator(args, cameras_train, poses_train, poses_train_2d,
                                                pad=pad, causal_shift=causal_shift, augment=False)
        print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
        if semi_supervised:
            print("Not Implement semi_supervised version for DDP")
            raise NotImplementedError

        if args.resume:
            chk_filename = os.path.join(args.checkpoint, args.resume)
            print("resuming the training...")
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=loc)
            epoch = checkpoint['epoch']
            model_pos_train.load_state_dict(checkpoint['model_pos'])
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
                # train_generator.set_random_state(checkpoint['random_state'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            if checkpoint['amp'] is not None:
                amp.load_state_dict(checkpoint['amp'])
            if args.rank % ngpus_per_node == 0:
                if 'train_global_steps' in checkpoint and 'valid_global_steps' in checkpoint:
                    writer_dict['train_global_steps'] = checkpoint['train_global_steps']
                    writer_dict['valid_global_steps'] = checkpoint['valid_global_steps']
            lr = checkpoint['lr']
            if semi_supervised:
                print("Not Implement semi_supervised version for DDP")
                raise NotImplementedError
                # model_traj_train.load_state_dict(checkpoint['model_traj'])
                # model_traj.load_state_dict(checkpoint['model_traj'])
                # semi_generator.set_random_state(checkpoint['random_state_semi'])
                
        logger.info('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
        logger.info('** The final evaluation will be carried out after the last training epoch.')
        
        myend = time()
        mytime = AverageMeter()
        best_valid = 50.0
        prof_flag = args.prof
        while epoch < args.epochs:
            start_time = time()
            epoch_loss = AverageMeter()
            epoch_loss_val = AverageMeter()
            train_generator.set_epoch(epoch)
            epoch_loss_3d_train = 0
            # epoch_loss_traj_train = 0
            # epoch_loss_2d_train_unlabeled = 0
            epoch_fps = AverageMeter()
            N = 0
            N_semi = 0
            model_pos_train.train()
            if semi_supervised:
                print("Not Implement semi_supervised version for DDP")
                raise NotImplementedError
            else:
                # Regular supervised scenario
                count = 0
                for _, batch_3d, batch_2d in train_generator.next_epoch():
                    if count >= 2:
                        my_epoch_start = time()
                    if batch_2d.shape[0] == 0:
                        continue
                    #     print(f"batch_3d.shape:{batch_3d.shape} for rank:{args.rank}")
                    bz = batch_2d.shape[0]
                    assert batch_3d.shape[0] == bz
                    inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.npu.is_available():
                        inputs_3d = inputs_3d.to(loc, non_blocking=False)
                        inputs_2d = inputs_2d.to(loc, non_blocking=False)
                    inputs_3d[:, :, 0] = 0

                    if prof_flag and count==10 and args.rank==0:
                        with torch.autograd.profiler.profile(use_npu=True) as prof:
                            optimizer.zero_grad()

                            # Predict 3D poses
                            predicted_3d_pos = model_pos_train(inputs_2d)
                            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)

                            loss_total = loss_3d_pos
                            if args.apex:
                                with amp.scale_loss(loss_total, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            else:
                                loss_total.backward()

                            optimizer.step()
                        print(prof.key_averages().table(sort_by='self_cpu_time_total'))
                        prof.export_chrome_trace(os.path.join(args.checkpoint,'out.prof'))
                        prof_flag = False
                        print(f"prof has been saved as {os.path.join(args.checkpoint,'out.prof')}")
                    else:
                        optimizer.zero_grad()

                        # Predict 3D poses
                        predicted_3d_pos = model_pos_train(inputs_2d)
                        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)

                        loss_total = loss_3d_pos
                        if args.apex:
                            with amp.scale_loss(loss_total, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss_total.backward()

                        optimizer.step()

                    dist.all_reduce(loss_total)
                    loss_total = loss_total / ngpus_per_node
                    epoch_loss.update(loss_total.item(), bz)

                    epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_total.item()
                    N += inputs_3d.shape[0]*inputs_3d.shape[1]

                    if count >= 2:
                        batch_time = time()-my_epoch_start
                        fps = bz * ngpus_per_node / batch_time
                        epoch_fps.update(fps)
                    if args.rank % ngpus_per_node == 0:
                        writer = writer_dict['writer']
                        train_step = writer_dict['train_global_steps']
                        writer.add_scalar('total_loss',epoch_loss.avg,train_step)
                        writer_dict['train_global_steps'] = train_step + 1

                    
                    if count % args.print_freq == 0 and args.rank % ngpus_per_node == 0:
                        logger.info("({batch}/{size})| loss:{loss.val:.5f} ({loss.avg:.5f})| FPS:{fps.val:.3f} ({fps.avg:.3f})".format(
                            batch=count, size=math.ceil(train_generator.num_frames()/(args.batch_size*ngpus_per_node)), loss=epoch_loss,
                            fps=epoch_fps
                        ))
                    count +=1
            if args.rank % ngpus_per_node == 0:
                writer.add_scalar('loss_3d/train', epoch_loss_3d_train / N, epoch)

            losses_3d_train.append(epoch_loss_3d_train / N)


            # End-of-epoch evaluation
            if args.rank == 0 and not args.no_eval:
                print("End of epoch evaluation start ....")
            with torch.no_grad():
                model_pos.load_state_dict(model_pos_train.state_dict())
                model_pos.eval()
                if semi_supervised:
                    print("Not Implement semi_supervised version for DDP")
                    raise NotImplementedError
                    # model_traj.load_state_dict(model_traj_train.state_dict())
                    # model_traj.eval()

                epoch_loss_3d_valid = 0
                epoch_loss_traj_valid = 0
                epoch_loss_2d_valid = 0
                N = 0
                
                if not args.no_eval:
                    # Evaluate on test set
                    for cam, batch, batch_2d in test_generator.next_epoch():
                        inputs_3d = torch.from_numpy(batch.astype('float32'))
                        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                        if torch.npu.is_available():
                            inputs_3d = inputs_3d.to(loc, non_blocking=False)
                            inputs_2d = inputs_2d.to(loc, non_blocking=False)
                        inputs_traj = inputs_3d[:, :, :1].clone()
                        inputs_3d[:, :, 0] = 0

                        # Predict 3D poses
                        predicted_3d_pos = model_pos(inputs_2d)
                        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)

                        bz = inputs_2d.shape[0]
                        assert bz == inputs_3d.shape[0]

                        dist.all_reduce(loss_3d_pos)
                        loss_3d_pos = loss_3d_pos / ngpus_per_node

                        epoch_loss_val.update(loss_3d_pos, bz)

                        epoch_loss_3d_valid += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                        N += inputs_3d.shape[0]*inputs_3d.shape[1]

                        if args.rank % ngpus_per_node == 0:
                            val_step = writer_dict['valid_global_steps']
                            writer.add_scalar("val_loss",epoch_loss_val.avg, val_step)
                            writer_dict['valid_global_steps'] = val_step + 1
                        if semi_supervised:
                            print("Not Implement semi_supervised version for DDP")
                            raise NotImplementedError
                    if args.rank % ngpus_per_node == 0:
                        writer.add_scalar("loss_3d/valid", epoch_loss_3d_valid / N, epoch)
                        print("out of end-of-epoch evaluation loop.")
                    losses_3d_valid.append(epoch_loss_3d_valid / N)
                    if semi_supervised:
                        print("Not Implement semi_supervised version for DDP")
                        raise NotImplementedError
                        # losses_traj_valid.append(epoch_loss_traj_valid / N)
                        # losses_2d_valid.append(epoch_loss_2d_valid / N)


                    # Evaluate on training set, this time in evaluation mode
                    epoch_loss_3d_train_eval = 0
                    # epoch_loss_traj_train_eval = 0
                    # epoch_loss_2d_train_labeled_eval = 0
                    N = 0
                    for cam, batch, batch_2d in train_generator_eval.next_epoch():
                        if batch_2d.shape[1] == 0:
                            # This can only happen when downsampling the dataset
                            continue
                            
                        inputs_3d = torch.from_numpy(batch.astype('float32'))
                        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                        if torch.npu.is_available():
                            inputs_3d = inputs_3d.npu()
                            inputs_2d = inputs_2d.npu()
                        inputs_traj = inputs_3d[:, :, :1].clone()
                        inputs_3d[:, :, 0] = 0

                        # Compute 3D poses
                        predicted_3d_pos = model_pos(inputs_2d)
                        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)

                        dist.all_reduce(loss_3d_pos)
                        loss_3d_pos = loss_3d_pos / ngpus_per_node
                        epoch_loss_3d_train_eval += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                        N += inputs_3d.shape[0]*inputs_3d.shape[1]

                        if semi_supervised:
                            print("Not Implement semi_supervised version for DDP")
                            raise NotImplementedError
                    
                    if args.rank % ngpus_per_node == 0:
                        writer.add_scalar('loss_3d/train_eval', epoch_loss_3d_train_eval / N, epoch)
                    losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)
                    if semi_supervised:
                        print("Not Implement semi_supervised version for DDP")
                        raise NotImplementedError
                        # losses_traj_train_eval.append(epoch_loss_traj_train_eval / N)
                        # losses_2d_train_labeled_eval.append(epoch_loss_2d_train_labeled_eval / N)

                    # Evaluate 2D loss on unlabeled training set (in evaluation mode)
                    epoch_loss_2d_train_unlabeled_eval = 0
                    N_semi = 0
                    if semi_supervised:
                        print("Not Implement semi_supervised version for DDP")
                        raise NotImplementedError

            elapsed = (time() - start_time)/60
            
            if args.rank % ngpus_per_node == 0:
                if args.no_eval:
                    logger.info('[%d] time %.2f lr %f 3d_train %f FPS %d' % (
                            epoch + 1,
                            elapsed,
                            lr,
                            losses_3d_train[-1] * 1000,
                            int(epoch_fps.avg)))
                else:
                    if semi_supervised:
                        print("Not Implement semi_supervised version for DDP")
                        raise NotImplementedError
                    else:
                        logger.info('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f FPS %d' % (
                                epoch + 1,
                                elapsed,
                                lr,
                                losses_3d_train[-1] * 1000,
                                losses_3d_train_eval[-1] * 1000,
                                losses_3d_valid[-1]  *1000,
                                int(epoch_fps.avg))
                            )
            
            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            epoch += 1
            
            # Decay BatchNorm momentum
            momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
            model_pos_train.module.set_bn_momentum(momentum)
            if semi_supervised:
                print("Not Implement semi_supervised version for DDP")
                raise NotImplementedError
                # model_traj_train.set_bn_momentum(momentum)
                
            # Save best valid
            if args.no_eval:
                valid = 0
            else:
                valid = losses_3d_valid[-1] *1000
            if args.rank % ngpus_per_node == 0 and valid < best_valid:
                best_valid = valid
                bst_path = os.path.join(args.checkpoint, 'model_best.bin')
                logger.info(f'Saving best model up to epoch:{epoch} to {bst_path}')
                torch.save({
                    'model_pos':model_pos_train.state_dict()
                }, bst_path)

            # Save checkpoint if necessary
            if epoch % args.checkpoint_frequency == 0 and args.rank % ngpus_per_node == 0:
                chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
                logger.info(f'Saving checkpoint to {chk_path}')
                
                torch.save({
                    'epoch': epoch,
                    'lr': lr,
                    # 'random_state': train_generator.random_state(),
                    'optimizer': optimizer.state_dict(),
                    'model_pos': model_pos_train.state_dict(),
                    # 'model_traj': None, # model_traj_train.state_dict() if semi_supervised else None,
                    'amp': amp.state_dict() if args.apex else None,
                    'random_state_semi': None, #semi_generator.random_state() if semi_supervised else None,
                    'train_global_steps': writer_dict['train_global_steps'],
                    'valid_global_steps': writer_dict['valid_global_steps']
                }, chk_path)
            
                
            # Save training curves after every epoch, as .png images (if requested)
            if args.export_training_curves and epoch > 3 and args.rank % ngpus_per_node == 0:
                if 'matplotlib' not in sys.modules:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                
                plt.figure()
                epoch_x = np.arange(3, len(losses_3d_train)) + 1
                plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
                plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
                plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
                plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
                plt.ylabel('MPJPE (m)')
                plt.xlabel('Epoch')
                plt.xlim((3, epoch))
                plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

                if semi_supervised:
                    print("Not Implement semi_supervised version for DDP")
                    raise NotImplementedError
                plt.close('all')

            mytime.update(time()-myend)
            myend = time()
            if args.rank % ngpus_per_node == 0:
                print(f"In average, it takes {time_format_convert(mytime.avg)} per epoch.")
                print(f"Time has elapsed {time_format_convert(mytime.sum)}")
                print(f"It still need {time_format_convert(mytime.avg*(args.epochs-epoch))}")
                print("****************************************************************************")  
    if args.rank % ngpus_per_node == 0:            
        writer_dict['writer'].close()
    # Evaluate
    if args.evaluate:
        logger.info('Evaluating...')
        # chk_filename = os.path.join(args.checkpoint, 'model_best.bin')
        # if (not args.evaluate) and (os.path.exists(chk_filename)):
        #     checkpoint = torch.load(chk_filename, map_location='cpu')
        #     model_pos.load_state_dict(checkpoint['model_pos'])
        all_actions = {}
        all_actions_by_subject = {}
        for subject in subjects_test:
            if subject not in all_actions_by_subject:
                all_actions_by_subject[subject] = {}

            for action in dataset[subject].keys():
                action_name = action.split(' ')[0]
                if action_name not in all_actions:
                    all_actions[action_name] = []
                if action_name not in all_actions_by_subject[subject]:
                    all_actions_by_subject[subject][action_name] = []
                all_actions[action_name].append((subject, action))
                all_actions_by_subject[subject][action_name].append((subject, action))

        if not args.by_subject:
            run_evaluation(args, all_actions, model_pos, None, keypoints, dataset, pad, causal_shift, kps_left, kps_right, joints_left, joints_right, action_filter)
        else:
            for subject in all_actions_by_subject.keys():
                if args.rank % ngpus_per_node == 0:
                    print('Evaluating on subject', subject)
                run_evaluation(args, all_actions_by_subject[subject], model_pos, None, keypoints, dataset, pad, causal_shift, kps_left, kps_right, joints_left, joints_right, action_filter)
                if args.rank % ngpus_per_node == 0:
                    print('')
    #dist.destroy_process_group()

if __name__ == "__main__":
    main()
