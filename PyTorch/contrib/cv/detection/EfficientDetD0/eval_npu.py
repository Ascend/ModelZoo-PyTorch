# Copyright 2021 Huawei Technologies Co., Ltd
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

#!/usr/bin/env python
""" EfficientDet Training Script

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by Ross Wightman (https://github.com/rwightman)

"""

import sys
import os
import argparse
import time
import yaml
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
if torch.__version__>="1.8":
    import torch_npu
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.npu.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

from effdet import create_model, unwrap_bench, create_loader, create_dataset, create_evaluator
from effdet.data import resolve_input_config, SkipSubset
from effdet.anchors import Anchors, AnchorLabeler
from timm.models import resume_checkpoint, load_checkpoint
from timm.models.layers import set_layer_config
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

torch.backends.cudnn.benchmark = True


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('--root', default='/home/sjh/datasets/coco',metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='coco', type=str, metavar='DATASET',
                    help='Name of model to train (default: "coco"')
parser.add_argument('--model', default='tf_efficientdet_d0', type=str, metavar='MODEL',
                    help='Name of model to train (default: "tf_efficientdet_d0"') 
add_bool_arg(parser, 'redundant-bias', default=None, help='override model config for redundant bias')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')   
parser.add_argument('--val-skip', type=int, default=0, metavar='N',
                    help='Skip every N validation samples.')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')   
parser.add_argument('--no-pretrained-backbone', action='store_true', default=True,
                    help='Do not start with pretrained backbone weights, fully random.')   
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', 
                    help='Resume full model and optimizer state from checkpoint (default: none)')  
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('-b', '--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 1)') 
parser.add_argument('--clip-grad', type=float, default=10.0, metavar='NORM',
                    help='Clip gradient norm (default: 10.0)')

# Optimizer parameters
parser.add_argument('--opt', default='fusedmomentum', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "fusedmomentum"')               
parser.add_argument('--opt-eps', default=1e-3, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=4e-5,
                    help='weight decay (default: 0.00004)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.32, metavar='LR',
                    help='learning rate (default: 0.09)')                     
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')                      
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')  
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', 
                    help='epochs to warmup LR, if scheduler supports') 
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')

# loss
parser.add_argument('--smoothing', type=float, default=None, help='override model config label smoothing')
add_bool_arg(parser, 'jit-loss', default=None, help='override model config for torchscript jit loss fn')
add_bool_arg(parser, 'legacy-focal', default=None, help='override model config to use legacy focal loss')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=True,     
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-decay', type=float, default=0.9999,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--sync-bn', action='store_true',default=True,
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')  
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=128, metavar='N',
                    help='how many training processes to use (default: 16 or 32×p)') 
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=True,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training') 
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.') 
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
add_bool_arg(parser, 'bench-labeler', default=False,
             help='label targets in model bench, increases GPU load at expense of loader processes')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='map', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "map"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')


parser.add_argument('--world-size', default=1, type=int,
                    help='.')                                             
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training.')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')                        
parser.add_argument('--addr', default='10.136.181.115', type=str, help='master addr')  ## --addr $(hostname -I |awk '{print $1}') \                      
parser.add_argument('--dist-url', default='tcp://127.0.0.1:50000', type=str,
                    help='url used to set up distributed training')      
parser.add_argument('--device-list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('--loss-scale', default= None, type=float,help='loss scale using in amp, default -1 means dynamic')  
parser.add_argument('--opt-level', default='O1', type=str,
                    help='loss scale using in amp, default -1 means dynamic, O1')   
parser.add_argument('--multiprocessing-distributed', action='store_true', default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dist-backend', default='hccl', type=str,
                    help='distributed backend')
# for servers to immediately record the logs
def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print
print = flush_print(print)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    # args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # return args, args_text
    return args


def main():
    setup_default_logging()
    args = _parse_args()
    
    # logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    #                 level=logging.DEBUG)

    def device_id_to_process_device_map(device_list):
        devices = device_list.split(",")
        devices = [int(x) for x in devices]
        devices.sort()

        process_device_map = dict()
        for process_id, device_id in enumerate(devices):
            process_device_map[process_id] = device_id

        return process_device_map
        
    os.environ['MASTER_ADDR'] = args.addr # 可以使用当前真实ip或者'127.0.0.1'
    os.environ['MASTER_PORT'] = '29688' # 随意一个可使用的port即可

    args.pretrained_backbone = not args.no_pretrained_backbone
    args.prefetcher = not args.no_prefetcher
    
    if "WORLD_SIZE" in os.environ: # 无这个环境
        args.world_size = int(os.environ["WORLD_SIZE"])
        print("[npu id:", args.gpu, "]","args.world_size",args.world_size)

    else:
        args.world_size = 1
        print("[npu id:", args.gpu, "]","args.world_size",args.world_size) 

    
    #args.distributed = False
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    
    args.process_device_map = device_id_to_process_device_map(args.device_list)
    if args.device == 'npu':
        ngpus_per_node = len(args.process_device_map) # 8
    else:
        ngpus_per_node = torch.cuda.device_count()
        
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        
        args.world_size = ngpus_per_node * args.world_size
        
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # The child process uses the environment variables of the parent process,
        # we have to set LOCAL_DEVICE_ID for every proc
        if args.device == 'npu':
            # main_worker(args.gpu, ngpus_per_node, args)
            print("before main_worker","gpu",args.gpu,"ngpus_per_node",ngpus_per_node) 
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    print("enter main_worker","gpu",gpu, args.gpu,"ngpus_per_node",ngpus_per_node) 
    if args.device_list != '':
        args.gpu = args.process_device_map[gpu]
    else:
        args.gpu = gpu

    os.environ['LOCAL_DEVICE_ID'] = str(args.gpu)
    print("[npu id:", args.gpu, "]", "++++++++++++++++ LOCAL_DEVICE_ID:", os.environ['LOCAL_DEVICE_ID']) 


    if args.gpu is not None:
        print("[npu id:", args.gpu, "]", "Use GPU: {} for training".format(args.gpu)) 

    if args.distributed:
        if args.rank == -1: # args.dist_url == "env://" and 
            args.rank = 0 #int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            # args.rank = int(os.environ["RANK"])
            print("[npu id:", args.gpu, "]","args.rank,ngpus_per_node,gpu",args.rank,ngpus_per_node,gpu)
            args.rank = args.rank * ngpus_per_node + gpu

        if args.device == 'npu':
            print("[npu id:", args.gpu, "]","npu init_process_group",args.world_size,args.rank,args.dist_backend) 
            dist.init_process_group(backend=args.dist_backend,  
                                    world_size=args.world_size, rank=args.rank)
            print("[npu id:", args.gpu, "]","done init_process_group",args.world_size,args.rank,args.dist_backend)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)

    loc = 'npu:{}'.format(args.gpu)
    torch.npu.set_device(loc)
    print("loc",loc)

    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    if args.distributed:
        print("[npu id:", args.gpu, "]",'Training in distributed mode with multiple processes, 1 NPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        print("[npu id:", args.gpu, "]",'Training with a single process on 1 NPU.')

    use_amp = 'apex'
    args.apex_amp = True

    print("[npu id:", args.gpu, "]", "===============main_worker()=================")
    print("[npu id:", args.gpu, "]", args)
    print("[npu id:", args.gpu, "]", "===============main_worker()=================")
    
    torch.manual_seed(args.seed + args.rank)

    with set_layer_config(scriptable=args.torchscript):
        model = create_model(
            args.model,
            bench_task='train',
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            pretrained_backbone=args.pretrained_backbone,
            redundant_bias=args.redundant_bias,
            label_smoothing=args.smoothing,
            legacy_focal=args.legacy_focal,
            jit_loss=args.jit_loss,
            soft_nms=args.soft_nms,
            bench_labeler=args.bench_labeler,
            checkpoint_path=args.initial_checkpoint,
        )
    model_config = model.config  # grab before we obscure with DP/DDP wrappers

    if args.rank == 0:
        print("[npu id:", args.gpu, "]",'Model %s created, param count: %d' % (args.model, sum([m.numel() for m in model.parameters()])))

    model = model.to(loc)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model, force native amp with `--native-amp` flag'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model. Use `--dist-bn reduce` instead of `--sync-bn`'
        model = torch.jit.script(model)

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale,combine_grad=True)
        loss_scaler = ApexScaler()
        if args.rank == 0:
            print("[npu id:", args.gpu, "]",'Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.npu.amp.autocast  
        loss_scaler = NativeScaler()
        if args.rank == 0:
            print("[npu id:", args.gpu, "]",'Using native Torch AMP. Training in mixed precision.')
    else:
        if args.rank == 0:
            print("[npu id:", args.gpu, "]",'AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            unwrap_bench(model), args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.rank == 0)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after npu(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
        if args.resume:
            load_checkpoint(unwrap_bench(model_ema), args.resume, use_ema=True)

    if args.distributed:
        if args.rank == 0:
            print("[npu id:", args.gpu, "]","Using torch DistributedDataParallel.")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
        # NOTE: EMA model does not need to be wrapped by DDP...
        if model_ema is not None and not args.resume:
            # ...but it is a good idea to sync EMA copy of weights
            # NOTE: ModelEma init could be moved after DDP wrapper if using PyTorch DDP, not Apex.
            model_ema.set(model)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.rank == 0:
        print("[npu id:", args.gpu, "]",'Scheduled epochs: {}'.format(num_epochs))

    loader_train, loader_eval, evaluator = create_datasets_and_loaders(args, model_config)

    if model_config.num_classes < loader_train.dataset.parser.max_label:
        logging.error(
            f'Model {model_config.num_classes} has fewer classes than dataset {loader_train.dataset.parser.max_label}.')
        exit(1)
    if model_config.num_classes > loader_train.dataset.parser.max_label:
        logging.warning(
            f'Model {model_config.num_classes} has more classes than dataset {loader_train.dataset.parser.max_label}.')

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = ''
    if args.rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            args.model
        ])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model, optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, decreasing=decreasing, unwrap_fn=unwrap_bench)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
            f.write(args_text)

    try:
        for epoch in range(start_epoch, start_epoch+1):   # num_epochs
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            print(torch.npu.synchronize(),"[npu id:", args.gpu, "]","one epoch suc")
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.rank == 0:
                    print("[npu id:", args.gpu, "]","Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            # the overhead of evaluating with coco style datasets is fairly high, so just ema or non, not both
            
            if model_ema is not None:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    print("[npu id:", args.gpu, "]","distribute_bn")
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                eval_metrics = validate(model_ema.module, loader_eval, args, evaluator, log_suffix=' (EMA)')
            else:
                eval_metrics = validate(model, loader_eval, args, evaluator)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def create_datasets_and_loaders(
        args,
        model_config,
        transform_train_fn=None,
        transform_eval_fn=None,
        collate_fn=None,
):
    """ Setup datasets, transforms, loaders, evaluator.

    Args:
        args: Command line args / config for training
        model_config: Model specific configuration dict / struct
        transform_train_fn: Override default image + annotation transforms (see note in loaders.py)
        transform_eval_fn: Override default image + annotation transforms (see note in loaders.py)
        collate_fn: Override default fast collate function

    Returns:
        Train loader, validation loader, evaluator
    """
    input_config = resolve_input_config(args, model_config=model_config)

    dataset_train, dataset_eval = create_dataset(args.dataset, args.root)

    # setup labeler in loader/collate_fn if not enabled in the model bench
    labeler = None
    if not args.bench_labeler:
        labeler = AnchorLabeler(
            Anchors.from_config(model_config), model_config.num_classes, match_threshold=0.5)

    loader_train = create_loader(
        dataset_train,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        # color_jitter=args.color_jitter,
        # auto_augment=args.aa,
        interpolation=args.train_interpolation or input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_train_fn,
        collate_fn=collate_fn,
    )
    print("[npu id:", args.gpu, "]","done loader_train")

    if args.val_skip > 1:
        dataset_eval = SkipSubset(dataset_eval, args.val_skip)
    loader_eval = create_loader(
        dataset_eval,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        anchor_labeler=labeler,
        transform_fn=transform_eval_fn,
        collate_fn=collate_fn,
    )
    print("[npu id:", args.gpu, "]","done loader_eval")

    evaluator = create_evaluator(args.dataset, loader_eval.dataset, distributed=args.distributed, pred_yxyx=False)

    print("[npu id:", args.gpu, "]","done evaluator") 

    return loader_train, loader_eval, evaluator


def train_epoch(
        epoch, model, loader, optimizer, args,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress, loss_scaler=None, model_ema=None):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.train()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    print("[npu id:", args.gpu, "]","num_updates",num_updates,len(loader))
    for batch_idx, (input, target) in enumerate(loader):
        
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        loc = 'npu:{}'.format(args.gpu)

        input = input.to(loc, non_blocking=True)
        for k,v in target.items():
            target[k] = v.to(loc, non_blocking=True)

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            output = model(input, target)
        loss = output['loss']

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters())
        else:
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        
        torch.npu.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if True: # args.local_rank == 0
                print("[npu id:", args.gpu, "]",
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/FPS  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/FPS)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, args, evaluator=None, log_suffix=''):   
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            loc = 'npu:{}'.format(args.gpu)   
            input = input.to(loc, non_blocking=True)
            for k,v in target.items():
                target[k] = v.to(loc, non_blocking=True)

            output = model(input, target)

            loss = output['loss']

            if evaluator is not None:
                evaluator.add_predictions(output['detections'], target)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            losses_m.update(reduced_loss.item(), input.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                print("[npu id:", args.gpu, "]",
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) {rate:>7.2f} '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                        log_name, batch_idx, last_idx, 
                        batch_time=batch_time_m, rate=input.size(0) * args.world_size / batch_time_m.val,
                        loss=losses_m))

    metrics = OrderedDict([('loss', losses_m.avg)])
    if evaluator is not None:
        metrics['map'] = evaluator.evaluate()
        print("[npu id:", args.gpu, "]",metrics['map'])

    return metrics

if __name__ == '__main__':
    main()
