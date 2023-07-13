"""
Copyright 2023 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import time
import argparse
import numpy as np
from contextlib import contextmanager

import torch
import torch_npu
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import models
import loss_functions
import data_functions
from tacotron2_common.utils import ParseFromConfigFile

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from scipy.io.wavfile import write as write_wav
from apex import amp

amp.lists.functional_overrides.FP32_FUNCS.remove('softmax')
amp.lists.functional_overrides.FP16_FUNCS.append('softmax')

CALCULATE_DEVICE = 'npu:0'


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('-m', '--model-name', type=str, default='', required=True,
                        help='Model to train')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--anneal-steps', nargs='*',
                        help='Epochs after which decrease learning rate')
    parser.add_argument('--anneal-factor', type=float, choices=[0.1, 0.3], default=0.1,
                        help='Factor for annealing learning rate')

    parser.add_argument('--config-file', action=ParseFromConfigFile,
                        config_type=str, config_help='Path to configuration file')

    parser.add_argument('--seed', type=int, help='set a seed of all randoms')
    parser.add_argument('--target_train_loss', type=float, default=0.57, help='set a target train loss')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, required=True,
                          help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=50,
                          help='Number of epochs per checkpoint')
    training.add_argument('--checkpoint-path', type=str, default='',
                          help='Checkpoint path to resume training')
    training.add_argument('--resume-from-last', action='store_true',
                          help='Resumes training from the last checkpoint; uses the directory provided with \'--output\' option to search for the checkpoint \"checkpoint_<model_name>_last.pt\"')
    training.add_argument('--dynamic-loss-scaling', type=bool, default=False,
                          help='Enable dynamic loss scaling')
    training.add_argument('--amp', action='store_true',
                          help='Enable AMP')
    training.add_argument('--cudnn-enabled', action='store_true',
                          help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', action='store_true',
                          help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument(
        '--use-saved-learning-rate', default=False, type=bool)
    optimization.add_argument('-lr', '--learning-rate', type=float, required=True,
                              help='Learing rate')
    optimization.add_argument('--weight-decay', default=1e-6, type=float,
                              help='Weight decay')
    optimization.add_argument('--grad-clip-thresh', default=1.0, type=float,
                              help='Clip threshold for gradients')
    optimization.add_argument('-bs', '--batch-size', type=int, required=True,
                              help='Batch size per GPU')
    optimization.add_argument('--grad-clip', default=5.0, type=float,
                              help='Enables gradient clipping and sets maximum gradient norm value')

    # dataset parameters
    dataset = parser.add_argument_group('dataset parameters')
    dataset.add_argument('--load-mel-from-disk', action='store_true',
                         help='Loads mel spectrograms from disk instead of computing them on the fly')
    dataset.add_argument('--training-files',
                         default='filelists/ljs_audio_text_train_filelist.txt',
                         type=str, help='Path to training filelist')
    dataset.add_argument('--validation-files',
                         default='filelists/ljs_audio_text_val_filelist.txt',
                         type=str, help='Path to validation filelist')
    dataset.add_argument('--text-cleaners', nargs='*',
                         default=['english_cleaners'], type=str,
                         help='Type of text cleaners for input text')

    # audio parameters
    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--max-wav-value', default=32768.0, type=float,
                       help='Maximum audiowave value')
    audio.add_argument('--sampling-rate', default=22050, type=int,
                       help='Sampling rate')
    audio.add_argument('--filter-length', default=1024, type=int,
                       help='Filter length')
    audio.add_argument('--hop-length', default=256, type=int,
                       help='Hop (stride) length')
    audio.add_argument('--win-length', default=1024, type=int,
                       help='Window length')
    audio.add_argument('--mel-fmin', default=0.0, type=float,
                       help='Minimum mel frequency')
    audio.add_argument('--mel-fmax', default=8000.0, type=float,
                       help='Maximum mel frequency')

    audio.add_argument('--mse_weight', default=1.0, type=float,
                       help='mes loss weight')
    audio.add_argument('--logits_weight', default=1.0, type=float,
                       help='logits loss weight')

    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='Rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int,
                             help='Number of processes, do not set! Done by multiproc module')
    distributed.add_argument('--dist-url', type=str, default='tcp://localhost:23456',
                             help='Url used to set up distributed training')
    distributed.add_argument('--group-name', type=str, default='group_name',
                             required=False, help='Distributed group name')
    distributed.add_argument('--dist-backend', default='hccl', type=str, choices={'nccl', 'hccl'},
                             help='Distributed run backend')

    benchmark = parser.add_argument_group('benchmark')
    benchmark.add_argument('--bench-class', type=str, default='')

    return parser


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    if rt.is_floating_point():
        rt = rt / num_gpus
    else:
        rt = rt // num_gpus
    return rt


def init_distributed(args, world_size, rank, group_name):
    assert torch.npu.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.npu.set_device(rank % torch.npu.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=world_size, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def save_checkpoint(model, optimizer, epoch, config, amp_run, output_dir, model_name,
                    local_rank, world_size):
    if local_rank == 0:
        checkpoint = {'epoch': epoch,
                      'config': config,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        if amp_run:
            checkpoint['amp'] = amp.state_dict()

        checkpoint_filename = "checkpoint_{}_{}.pt".format(model_name, epoch)
        checkpoint_path = os.path.join(output_dir, checkpoint_filename)
        print("Saving model and optimizer state at epoch {} to {}".format(
            epoch, checkpoint_path))
        torch.save(checkpoint, checkpoint_path)

        symlink_src = checkpoint_filename
        symlink_dst = os.path.join(
            output_dir, "checkpoint_{}_last.pt".format(model_name))
        if os.path.exists(symlink_dst) and os.path.islink(symlink_dst):
            print("Updating symlink", symlink_dst, "to point to", symlink_src)
            os.remove(symlink_dst)

        os.symlink(symlink_src, symlink_dst)


def get_last_checkpoint_filename(output_dir, model_name):
    symlink = os.path.join(output_dir, "checkpoint_{}_last.pt".format(model_name))
    if os.path.exists(symlink):
        print("Loading checkpoint from symlink", symlink)
        return os.path.join(output_dir, os.readlink(symlink))
    else:
        print("No last checkpoint available - starting from epoch 0 ")
        return ""


def load_checkpoint(model, optimizer, epoch, config, amp_run, filepath, local_rank):
    checkpoint = torch.load(filepath, map_location='cpu')

    epoch[0] = checkpoint['epoch'] + 1
    device_id = local_rank % torch.npu.device_count()
    torch.npu.set_rng_state(checkpoint['cuda_rng_state_all'][device_id])
    if 'random_rng_states_all' in checkpoint:
        torch.random.set_rng_state(checkpoint['random_rng_states_all'][device_id])
    elif 'random_rng_state' in checkpoint:
        torch.random.set_rng_state(checkpoint['random_rng_state'])
    else:
        raise Exception("Model checkpoint must have either 'random_rng_state' or 'random_rng_states_all' key.")
    config = checkpoint['config']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if amp_run:
        amp.load_state_dict(checkpoint['amp'])


@contextmanager
def evaluating(model):
    '''Temporarily switch to evaluation mode.'''
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()


def validate(model, criterion, valset, epoch, batch_iter, batch_size,
             world_size, collate_fn, distributed_run, rank, batch_to_gpu, summ_writter):
    """Handles all the validation scoring and printing"""
    with evaluating(model), torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=0, shuffle=False,
                                sampler=val_sampler,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn)

        val_loss = 0.0
        num_iters = 0
        val_items_per_sec = 0.0
        for i, batch in enumerate(val_loader):
            iter_start_time = time.perf_counter()

            x, y, num_items = batch_to_gpu(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, world_size).item()
                reduced_num_items = reduce_tensor(num_items.data, 1).item()
            else:  #
                reduced_val_loss = loss.item()
                reduced_num_items = num_items.item()
            val_loss += reduced_val_loss

            iter_stop_time = time.perf_counter()
            iter_time = iter_stop_time - iter_start_time

            items_per_sec = reduced_num_items / iter_time
            DLLogger.log(step=(epoch, batch_iter, i), data={'val_items_per_sec': items_per_sec})
            val_items_per_sec += items_per_sec
            num_iters += 1

        val_loss = val_loss / (i + 1)

        DLLogger.log(step=(epoch,), data={'val_loss': val_loss})
        DLLogger.log(step=(epoch,), data={'val_items_per_sec':
                                              (val_items_per_sec / num_iters if num_iters > 0 else 0.0)})
        summ_writter.add_scalar('val_loss/loss', val_loss, epoch)

        return val_loss, val_items_per_sec


def adjust_learning_rate(iteration, epoch, optimizer, learning_rate,
                         anneal_steps, anneal_factor, rank):
    p = 0
    if anneal_steps is not None:
        for i, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p + 1

    if anneal_factor == 0.3:
        lr = learning_rate * ((0.1 ** (p // 2)) * (1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate * (anneal_factor ** p)

    if optimizer.param_groups[0]['lr'] != lr:
        DLLogger.log(step=(epoch, iteration),
                     data={'learning_rate changed': str(optimizer.param_groups[0]['lr']) + " -> " + str(lr)})

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Training')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    if args.seed is not None:
        seed_everything(args.seed)

    summ_writer = SummaryWriter('./tb')

    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ["RANK_SIZE"])
    else:
        local_rank = args.rank
        world_size = args.world_size

    distributed_run = world_size > 1

    if local_rank == 0:
        log_file = os.path.join(args.output, args.log_file)
        DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_file),
                                StdOutBackend(Verbosity.VERBOSE)])
    else:
        DLLogger.init(backends=[])

    for k, v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k: v})
    DLLogger.log(step="PARAMETER", data={'model_name': 'Tacotron2_PyT'})

    model_name = args.model_name
    parser = models.model_parser(model_name, parser)
    args, _ = parser.parse_known_args()

    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if distributed_run:
        init_distributed(args, world_size, local_rank, args.group_name)
    else:
        torch.npu.set_device(CALCULATE_DEVICE)

    run_start_time = time.perf_counter()

    model_config = models.get_model_config(model_name, args)
    model = models.get_model(model_name, model_config,
                             cpu_run=False,
                             uniform_initialize_bn_weight=not args.disable_uniform_initialize_bn_weight)

    if not args.amp and distributed_run:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=128)
        if distributed_run:
            model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    try:
        sigma = args.sigma
    except AttributeError:
        sigma = None

    start_epoch = [0]

    if args.resume_from_last:
        args.checkpoint_path = get_last_checkpoint_filename(args.output, model_name)

    if args.checkpoint_path is not "":
        load_checkpoint(model, optimizer, start_epoch, model_config,
                        args.amp, args.checkpoint_path, local_rank)

    start_epoch = start_epoch[0]

    criterion = loss_functions.get_loss_function(model_name, args, sigma)

    try:
        n_frames_per_step = args.n_frames_per_step
    except AttributeError:
        n_frames_per_step = None

    collate_fn = data_functions.get_collate_function(
        model_name, n_frames_per_step)
    trainset = data_functions.get_data_loader(
        model_name, args.dataset_path, args.training_files, args)
    if distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=2, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=args.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)

    valset = data_functions.get_data_loader(
        model_name, args.dataset_path, args.validation_files, args)

    batch_to_gpu = data_functions.get_batch_to_gpu(model_name)

    iteration = 0
    train_epoch_items_per_sec = 0.0
    val_loss = 0.0
    num_iters = 0

    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.perf_counter()
        # used to calculate avg items/sec over epoch
        reduced_num_items_epoch = 0

        train_epoch_items_per_sec = 0.0

        num_iters = 0
        reduced_loss = 0

        # if overflow at the last iteration then do not save checkpoint
        overflow = False

        if distributed_run:
            train_loader.sampler.set_epoch(epoch)
        iter_stop_time = time.perf_counter()

        for i, batch in enumerate(train_loader):

            iter_start_time = time.perf_counter()
            DLLogger.log(step=(epoch, i),
                         data={'data time': str(iter_start_time - iter_stop_time)})
            DLLogger.log(step=(epoch, i),
                         data={'glob_iter/iters_per_epoch': str(iteration) + "/" + str(len(train_loader))})

            adjust_learning_rate(iteration, epoch, optimizer, args.learning_rate,
                                 args.anneal_steps, args.anneal_factor, local_rank)

            model.zero_grad()

            x, y, num_items = batch_to_gpu(batch)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            if distributed_run:
                reduced_loss = reduce_tensor(loss.data, world_size).item()
                reduced_num_items = reduce_tensor(num_items.data, 1).item()
            else:
                reduced_loss = loss.item()
                reduced_num_items = num_items.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            num_iters += 1

            # accumulate number of items processed in this epoch
            reduced_num_items_epoch += reduced_num_items
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.grad_clip_thresh)
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_thresh)

            optimizer.step()

            stream = torch.npu.current_stream()
            stream.synchronize()

            iter_stop_time = time.perf_counter()
            iter_time = iter_stop_time - iter_start_time
            items_per_sec = reduced_num_items / iter_time
            train_epoch_items_per_sec += items_per_sec

            DLLogger.log(step=(epoch, i), data={'train_loss': reduced_loss})
            summ_writer.add_scalar('train_loss/loss', reduced_loss, iteration)
            DLLogger.log(step=(epoch, i), data={'train_items_per_sec': items_per_sec})
            DLLogger.log(step=(epoch, i), data={'train_iter_time': iter_time})
            iteration += 1

        epoch_stop_time = time.perf_counter()
        epoch_time = epoch_stop_time - epoch_start_time

        DLLogger.log(step=(epoch,), data={'train_items_per_sec':
                                              (train_epoch_items_per_sec / num_iters if num_iters > 0 else 0.0)})
        DLLogger.log(step=(epoch,), data={'train_loss': reduced_loss})
        DLLogger.log(step=(epoch,), data={'train_epoch_time': epoch_time})

        val_loss, val_items_per_sec = validate(model, criterion, valset, epoch,
                                               iteration, args.batch_size,
                                               world_size, collate_fn,
                                               distributed_run, local_rank,
                                               batch_to_gpu, summ_writer)

        if (epoch % args.epochs_per_checkpoint == 0) and args.bench_class == "":
            save_checkpoint(model, optimizer, epoch, model_config,
                            args.amp, args.output, args.model_name,
                            local_rank, world_size)
        if local_rank == 0:
            DLLogger.flush()
        if reduced_loss < args.target_train_loss:
            exit('train end')

    run_stop_time = time.perf_counter()
    run_time = run_stop_time - run_start_time
    DLLogger.log(step=tuple(), data={'run_time': run_time})
    DLLogger.log(step=tuple(), data={'val_loss': val_loss})
    DLLogger.log(step=tuple(), data={'train_items_per_sec':
                                         (train_epoch_items_per_sec / num_iters if num_iters > 0 else 0.0)})
    DLLogger.log(step=tuple(), data={'val_items_per_sec': val_items_per_sec})

    if local_rank == 0:
        DLLogger.flush()


# define hook
def hook_func(name, save_dict, module):
    def hook_function(module, inputs, outputs):
        input_key = name + ' inputs'
        idx = 1
        while input_key in save_dict:
            input_key = input_key.split('_')[0] + '_%d' % idx
            idx += 1
        save_dict[input_key] = inputs

        output_key = name + ' outputs'
        idx = 1
        while output_key in save_dict:
            output_key = output_key.split('_')[0] + '_%d' % idx
            idx += 1
        save_dict[output_key] = outputs

    return hook_function


def set_device(obj, device='cpu'):
    if isinstance(obj, (tuple, list)):
        dump = []
        for item in obj:
            dump.append(set_device(item, device))
        return dump
    elif isinstance(obj, dict):
        dump = {}
        for k, v in obj.items():
            dump[k] = set_device(v, device)
        return dump
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


def dump_tensor(output, name):
    dump = set_device(output, 'cpu')
    torch.save(dump, name)
    print('%s dump success!' % (name))


def load_tensor(name, device):
    output = torch.load(name)
    dump = set_device(output, device)
    print('%s load success!' % (name))
    return dump


if __name__ == '__main__':
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option['NPU_FUZZY_COMPILE_BLACKLIST'] = 'DynamicRNN,DynamicRNNV2'
    torch.npu.set_option(option)
    main()
