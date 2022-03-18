# Copyright 2020 Huawei Technologies Co., Ltd
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
import models
import argparse
import copy
import glob
import os
import re
import time
import warnings
from collections import defaultdict, OrderedDict

try:
    import nvidia_dlprof_pytorch_nvtx as pyprof
except ModuleNotFoundError:
    try:
        import pyprof
    except ModuleNotFoundError:
        warnings.warn('PyProf is unavailable')

import numpy as np
import torch
import torch.distributed as dist
from apex import amp
from apex.optimizers import NpuFusedAdam, NpuFusedLamb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from common.text import cmudict
from common.utils import prepare_tmp
from fastpitch.attn_loss_function import AttentionBinarizationLoss
from fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
from fastpitch.loss_function import FastPitchLoss


def parse_args(parser):
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')
    # parser.add_argument('--log-file', type=str, default=None,
    #                     help='Path to a DLLogger log file')
    parser.add_argument('--pyprof', action='store_true',
                        help='Enable pyprof profiling')

    train = parser.add_argument_group('training setup')
    train.add_argument('--epochs', type=int, required=True,
                       help='Number of total epochs to run')
    train.add_argument('--epochs-per-checkpoint', type=int, default=50,
                       help='Number of epochs per checkpoint')
    train.add_argument('--checkpoint-path', type=str, default=None,
                       help='Checkpoint path to resume training')
    train.add_argument('--resume', action='store_true',
                       help='Resume training from the last checkpoint')
    train.add_argument('--seed', type=int, default=1234,
                       help='Seed for PyTorch random number generators')
    train.add_argument('--amp', action='store_true',
                       help='Enable AMP')
    train.add_argument('--cuda', action='store_true',
                       help='Run on GPU using CUDA')
    train.add_argument('--cudnn-benchmark', action='store_true',
                       help='Enable cudnn benchmark mode')
    train.add_argument('--ema-decay', type=float, default=0,
                       help='Discounting factor for training weights EMA')
    train.add_argument('--grad-accumulation', type=int, default=1,
                       help='Training steps to accumulate gradients for')
    train.add_argument('--kl-loss-start-epoch', type=int, default=250,
                       help='Start adding the hard attention loss term')
    train.add_argument('--kl-loss-warmup-epochs', type=int, default=100,
                       help='Gradually increase the hard attention loss term')
    train.add_argument('--kl-loss-weight', type=float, default=1.0,
                       help='Gradually increase the hard attention loss term')

    opt = parser.add_argument_group('optimization setup')
    opt.add_argument('-lr', '--learning-rate', type=float, required=True,
                     help='Learing rate')
    opt.add_argument('--weight-decay', default=1e-6, type=float,
                     help='Weight decay')
    opt.add_argument('--grad-clip-thresh', default=1000.0, type=float,
                     help='Clip threshold for gradients')
    opt.add_argument('-bs', '--batch-size', type=int, required=True,
                     help='Batch size per GPU')
    opt.add_argument('--warmup-steps', type=int, default=1000,
                     help='Number of steps for lr warmup')
    opt.add_argument('--dur-predictor-loss-scale', type=float,
                     default=1.0, help='Rescale duration predictor loss')
    opt.add_argument('--pitch-predictor-loss-scale', type=float,
                     default=1.0, help='Rescale pitch predictor loss')
    opt.add_argument('--attn-loss-scale', type=float,
                     default=1.0, help='Rescale alignment loss')

    data = parser.add_argument_group('dataset parameters')
    data.add_argument('--training-files', type=str, nargs='*', required=True,
                      help='Paths to training filelists.')
    data.add_argument('--validation-files', type=str, nargs='*',
                      required=True, help='Paths to validation filelists')
    data.add_argument('--text-cleaners', nargs='*',
                      default=['english_cleaners'], type=str,
                      help='Type of text cleaners for input text')
    data.add_argument('--symbol-set', type=str, default='english_basic',
                      help='Define symbol set for input text')
    data.add_argument('--p-arpabet', type=float, default=0.0,
                      help='Probability of using arpabets instead of graphemes '
                           'for each word; set 0 for pure grapheme training')
    data.add_argument('--heteronyms-path', type=str, default='cmudict/heteronyms',
                      help='Path to the list of heteronyms')
    data.add_argument('--cmudict-path', type=str, default='cmudict/cmudict-0.7b',
                      help='Path to the pronouncing dictionary')
    data.add_argument('--prepend-space-to-text', action='store_true',
                      help='Capture leading silence with a space token')
    data.add_argument('--append-space-to-text', action='store_true',
                      help='Capture trailing silence with a space token')

    cond = parser.add_argument_group('data for conditioning')
    cond.add_argument('--n-speakers', type=int, default=1,
                      help='Number of speakers in the dataset. '
                           'n_speakers > 1 enables speaker embeddings')
    cond.add_argument('--load-pitch-from-disk', action='store_true',
                      help='Use pitch cached on disk with prepare_dataset.py')
    cond.add_argument('--pitch-online-method', default='praat',
                      choices=['praat', 'pyin'],
                      help='Calculate pitch on the fly during trainig')
    cond.add_argument('--pitch-online-dir', type=str, default=None,
                      help='A directory for storing pitch calculated on-line')
    cond.add_argument('--pitch-mean', type=float, default=214.72203,
                      help='Normalization value for pitch')
    cond.add_argument('--pitch-std', type=float, default=65.72038,
                      help='Normalization value for pitch')
    cond.add_argument('--load-mel-from-disk', action='store_true',
                      help='Use mel-spectrograms cache on the disk')  # XXX

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

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0),
                      help='Rank of the process for multiproc; do not set manually')
    dist.add_argument('--world_size', type=int, default=os.getenv('WORLD_SIZE', 1),
                      help='Number of processes for multiproc; do not set manually')
    dist.add_argument('--num_workers', type=int, default=4,
                      help='num_workers')
    dist.add_argument('--dist_backend', type=str, default='hccl')
    return parser


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)


def init_distributed(args, world_size, rank):
    assert torch.npu.is_available(), "Distributed mode requires NPU."
    print("Initializing distributed training")

    # Set npu device so everything is done on the right NPU.
    torch.npu.set_device(rank % torch.npu.device_count())

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29688'
    # Initialize distributed communication
    dist.init_process_group(backend=args.dist_backend, 
                    world_size=world_size, rank=args.local_rank)
    print("Done initializing distributed training", args.local_rank)


def last_checkpoint(output):
    def corrupted(fpath):
        try:
            torch.load(fpath, map_location='cpu')
            return False
        except:
            warnings.warn(f'Cannot load {fpath}')
            return True

    saved = sorted(
        glob.glob(f'{output}/FastPitch_checkpoint_*.pt'),
        key=lambda f: int(re.search('_(\d+).pt', f).group(1)))

    if len(saved) >= 1 and not corrupted(saved[-1]):
        return saved[-1]
    elif len(saved) >= 2:
        return saved[-2]
    else:
        return None


def maybe_save_checkpoint(args, model, ema_model, optimizer, epoch,
                          total_iter, config, final_checkpoint=False):
    if args.local_rank != 0:
        return

    intermediate = (args.epochs_per_checkpoint > 0
                    and epoch % args.epochs_per_checkpoint == 0)

    if not intermediate and epoch < args.epochs:
        return

    fpath = os.path.join(args.output, f"FastPitch_checkpoint_{epoch}.pt")
    print(f"Saving model and optimizer state at epoch {epoch} to {fpath}")
    ema_dict = None if ema_model is None else ema_model.state_dict()
    checkpoint = {'epoch': epoch,
                  'iteration': total_iter,
                  'config': config,
                  'state_dict': model.state_dict(),
                  'ema_state_dict': ema_dict,
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, fpath)


def load_checkpoint(args, model, ema_model, optimizer, epoch,
                    total_iter, config, filepath):
    if args.local_rank == 0:
        print(f'Loading model and optimizer state from {filepath}')
    checkpoint = torch.load(filepath, map_location='cpu')
    epoch[0] = checkpoint['epoch'] + 1
    total_iter[0] = checkpoint['iteration']

    sd = {k.replace('module.', ''): v
          for k, v in checkpoint['state_dict'].items()}
    getattr(model, 'module', model).load_state_dict(sd)
    optimizer.load_state_dict(checkpoint['optimizer'])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])


def validate(args, model, epoch, total_iter, criterion, valset,
             collate_fn, distributed_run, batch_to_gpu, ema=False):
    """Handles all the validation scoring and printing"""
    was_training = model.training
    model.eval()

    tik = time.perf_counter()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                                sampler=val_sampler,
                                batch_size=args.batch_size, pin_memory=False,
                                collate_fn=collate_fn)
        val_meta = defaultdict(float)
        val_num_frames = 0
        for i, batch in enumerate(val_loader):
            x, y, num_frames = batch_to_gpu(batch)
            y_pred = model(x)
            loss, meta = criterion(y_pred, y, is_training=False, meta_agg='sum')

            if distributed_run:
                for k, v in meta.items():
                    val_meta[k] += reduce_tensor(v, 1)
                val_num_frames += reduce_tensor(num_frames.data, 1).item()
            else:
                for k, v in meta.items():
                    val_meta[k] += v
                val_num_frames = num_frames.item()

        val_meta = {k: v / len(valset) for k, v in val_meta.items()}

    val_meta['took'] = time.perf_counter() - tik


    if args.local_rank == 0:
        print(f"epoch {epoch}|avg val loss:{val_meta['loss'].item():.2f}|avg val mel loss:{val_meta['mel_loss'].item():.2f}|{num_frames.item() / val_meta['took']:.2f}frames/s |took {val_meta['took']:.2f} s")

    if was_training:
        model.train()
    return val_meta


def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate * scale


def main():
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Training',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    print(args)

    if args.p_arpabet > 0.0:
        cmudict.initialize(args.cmudict_path, keep_ambiguous=True)

    distributed_run = args.world_size > 1

    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)

    if args.local_rank == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output)


    parser = models.parse_model_args('FastPitch', parser)
    args, unk_args = parser.parse_known_args()

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if distributed_run:
        init_distributed(args, args.world_size, args.local_rank)

    device = torch.device('npu' if args.cuda else 'cpu')
    model_config = models.get_model_config('FastPitch', args)
    model = models.get_model('FastPitch', model_config, device)
    
    attention_kl_loss = AttentionBinarizationLoss()

    # Store pitch mean/std as params to translate from Hz during inference
    model.pitch_mean[0] = args.pitch_mean
    model.pitch_std[0] = args.pitch_std

    kw = dict(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9,
              weight_decay=args.weight_decay)
    optimizer = NpuFusedLamb(model.parameters(), **kw)
    amp.register_half_function(torch, 'bmm') 
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=128.0, combine_grad=True)

    ema_model = None

    if distributed_run:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    start_epoch = [1]
    start_iter = [0]

    assert args.checkpoint_path is None or args.resume is False, (
        "Specify a single checkpoint source")
    if args.checkpoint_path is not None:
        ch_fpath = args.checkpoint_path
    elif args.resume:
        ch_fpath = last_checkpoint(args.output)
    else:
        ch_fpath = None

    if ch_fpath is not None:
        load_checkpoint(args, model, ema_model, optimizer,
                       start_epoch, start_iter, model_config, ch_fpath)

    start_epoch = start_epoch[0]
    total_iter = start_iter[0]
    
    criterion = FastPitchLoss(
        dur_predictor_loss_scale=args.dur_predictor_loss_scale,
        pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
        attn_loss_scale=args.attn_loss_scale)

    collate_fn = TTSCollate()

    if args.local_rank == 0:
        prepare_tmp(args.pitch_online_dir)

    trainset = TTSDataset(audiopaths_and_text=args.training_files, **vars(args))
    valset = TTSDataset(audiopaths_and_text=args.validation_files, **vars(args))

    if distributed_run:
        train_sampler, shuffle = DistributedSampler(trainset), False
    else:
        train_sampler, shuffle = None, True

    # 4 workers are optimal on DGX-1 (from epoch 2 onwards)
    train_loader = DataLoader(trainset, num_workers=args.num_workers, shuffle=shuffle,
                              sampler=train_sampler, batch_size=args.batch_size,
                              pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)

    model.train()
    
    epoch_loss = []
    epoch_mel_loss = []
    epoch_num_frames = []
    epoch_frames_per_sec = []
    epoch_time = []

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()

        epoch_loss += [0.0]
        epoch_mel_loss += [0.0]
        epoch_num_frames += [0]
        epoch_frames_per_sec += [0.0]

        if distributed_run:
            train_loader.sampler.set_epoch(epoch)

        accumulated_steps = 0
        iter_loss = 0
        iter_num_frames = 0
        iter_meta = {}
        iter_start_time = None

        epoch_iter = 0
        num_iters = len(train_loader) // args.grad_accumulation


        for batch in train_loader:
            if accumulated_steps == 0:
                if epoch_iter == num_iters:
                    break
                total_iter += 1
                epoch_iter += 1
                if iter_start_time is None:
                    iter_start_time = time.perf_counter()

                adjust_learning_rate(total_iter, optimizer, args.learning_rate,
                                    args.warmup_steps)

                optimizer.zero_grad()

            x, y, num_frames = batch_to_gpu(batch)
            
            y_pred = model(x)
            loss, meta = criterion(y_pred, y)

            if (args.kl_loss_start_epoch is not None and epoch >= args.kl_loss_start_epoch):
                if args.kl_loss_start_epoch == epoch and epoch_iter == 1:
                    print('Begin hard_attn loss')
                _, _, _, _, _, _, _, _, attn_soft, attn_hard, _, _ = y_pred
                binarization_loss = attention_kl_loss(attn_hard, attn_soft)
                kl_weight = min((epoch - args.kl_loss_start_epoch) / args.kl_loss_warmup_epochs, 1.0) * args.kl_loss_weight
                meta['kl_loss'] = binarization_loss.clone().detach() * kl_weight
                loss += kl_weight * binarization_loss
            else:
                meta['kl_loss'] = torch.zeros_like(loss)
                kl_weight = 0
                binarization_loss = 0
                
            loss /= args.grad_accumulation

            meta = {k: v / args.grad_accumulation
                    for k, v in meta.items()}
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if distributed_run:
                reduced_loss = reduce_tensor(loss.data, args.world_size).item()
                reduced_num_frames = reduce_tensor(num_frames.data, 1).item()
                meta = {k: reduce_tensor(v, args.world_size) for k, v in meta.items()}
            else:
                reduced_loss = loss.item()
                reduced_num_frames = num_frames.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            accumulated_steps += 1
            iter_loss += reduced_loss
            iter_num_frames += reduced_num_frames
            iter_meta = {k: iter_meta.get(k, 0) + meta.get(k, 0) for k in meta}
            
            if accumulated_steps % args.grad_accumulation == 0:

                if args.amp:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_thresh)
                    optimizer.step()
                    
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_thresh)
                    optimizer.step()


                iter_time = time.perf_counter() - iter_start_time
                iter_mel_loss = iter_meta['mel_loss'].item()
                iter_kl_loss = iter_meta['kl_loss'].item()
                epoch_frames_per_sec[-1] += iter_num_frames / iter_time
                epoch_loss[-1] += iter_loss
                epoch_num_frames[-1] += iter_num_frames
                epoch_mel_loss[-1] += iter_mel_loss
                
                if args.local_rank == 0:
                    print(f"epoch {epoch}|iter{epoch_iter}/{num_iters}|loss:{iter_loss:.2f}|mel_loss:{iter_mel_loss:.2f}|kl_loss:{iter_kl_loss:.5f}|kl_weight:{kl_weight:.3f}|{iter_num_frames / iter_time:.2f}frames/s|took {iter_time:.2f}s|lrate:{optimizer.param_groups[0]['lr']:3e}")
                
                if args.epochs == 1 and epoch_iter == 30:
                    exit()
                    
                accumulated_steps = 0
                iter_loss = 0
                iter_num_frames = 0
                iter_meta = {}
                iter_start_time = time.perf_counter()

        # Finished epoch
        epoch_loss[-1] /= epoch_iter
        epoch_mel_loss[-1] /= epoch_iter
        epoch_time += [time.perf_counter() - epoch_start_time]
        iter_start_time = None

        if args.local_rank == 0:
            print(f"epoch {epoch}|avg train loss:{epoch_loss[-1]:.2f}|avg train mel loss:{epoch_mel_loss[-1]:.2f}|{epoch_num_frames[-1] / epoch_time[-1]:.2f} frames/s|took {epoch_time[-1]:.2f}s")
        
        maybe_save_checkpoint(args, model, ema_model, optimizer, epoch,
                              total_iter, model_config)
        validate(args, model, epoch, total_iter, criterion, valset,
                 collate_fn, distributed_run, batch_to_gpu)


    if len(epoch_loss) > 0:
        # Was trained - average the last 20 measurements
        last_ = lambda l: np.asarray(l[-20:])
        epoch_loss = last_(epoch_loss)
        epoch_mel_loss = last_(epoch_mel_loss)
        epoch_num_frames = last_(epoch_num_frames)
        epoch_time = last_(epoch_time)
    print(f"avg train loss:{epoch_loss.mean()}|avg train mel loss:{epoch_mel_loss.mean()}|{epoch_num_frames.sum() / epoch_time.sum()} frames/s|took {epoch_time.mean()}s")

    validate(args, model, None, total_iter, criterion, valset,
             collate_fn, distributed_run, batch_to_gpu)


if __name__ == '__main__':
    main()
