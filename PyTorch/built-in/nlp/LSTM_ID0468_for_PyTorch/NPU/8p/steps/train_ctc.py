# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/usr/bin/python
# encoding=utf-8
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import copy
import time
import yaml
import argparse
import random
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.npu
import torch.multiprocessing
import torch.distributed as dist
import numpy as np
import apex
from apex import amp

sys.path.append('./')
from models.model_ctc import *
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader

supported_rnn = {'nn.LSTM': nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN': nn.RNN}
supported_activate = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURRENT_PATH, '../../../url.ini', 'r')) as _f:
    content = _f.read()
    master_url = content.split('master_url=')[1].split('\n')[0]
    
parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
parser.add_argument('--conf', default='conf/ctc_config.yaml', help='conf file with argument of LSTM and training')

parser.add_argument('-j', '--workers', default=128, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')

parser.add_argument('--use_npu', default=True, type=str, help='use npu to train the model')
parser.add_argument('--device_list', default='0,1,2,3,4,5,6,7', type=str, help='device id list')
parser.add_argument('--device', default='npu', type=str, help='npu or gpu')
# apex
parser.add_argument('--amp', default=False, action='store_true',
                    help='use amp to train the model')
parser.add_argument('--device_id', default=5, type=int, help='device_id')
parser.add_argument('--apex', action='store_true',
                                       help='User apex for mixed precision training')
parser.add_argument('--loss_scale', default=128., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt_level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--addr', default=master_url, type=str, help='master addr')


MAX = 2147483647


def _gen_seeds(shape):
    return np.random.uniform(1, MAX, size=shape).astype(np.float32)


seed_shape = (32 * 1024 * 12,)


def device_id_to_process_device_map(device_list):
    devices = device_list.split(",")
    devices = [int(x) for x in devices]
    devices.sort()
    process_device_map = dict()
    for process_id, device_id in enumerate(devices):
        process_device_map[process_id] = device_id

    return process_device_map


def run_epoch(epoch_id, model, data_iter, loss_fn, device, args, opts, sum_writer, optimizer=None, print_every=20,
              is_training=True):
    if is_training:
        model.train()
    else:
        model.eval()
    batch_time = 0
    data_time = 0
    total_loss = 0
    total_tokens = 0
    total_errs = 0
    cur_loss = 0
    i = 0
    steps_per_epoch = len(data_iter)
    end = time.time()
    for i, data in enumerate(data_iter):
        if i == 4:
            batch_time = 0
            data_time = 0
        data_time += (time.time() - end)
        global_step = (epoch_id - 1) * steps_per_epoch + i
        inputs, input_sizes, targets, target_sizes, utt_list = data

        inputs_npu = inputs.to(device)
        input_sizes_npu = input_sizes.to(device)
        targets_npu = targets.to(device)
        target_sizes_npu = target_sizes.to(device)

        out = model(inputs_npu)
        out_len, batch_size, _ = out.size()
        input_sizes_npu = (input_sizes_npu * out_len).long()

        loss = loss_fn(out, targets_npu, input_sizes_npu, target_sizes_npu)
        prob, index = torch.max(out, dim=-1)
        loss /= batch_size
        cur_loss += loss.item()
        total_loss += loss.item()
        index = index.cpu().transpose(0, 1)
        input_sizes = input_sizes_npu.cpu()

        if is_training:
            optimizer.zero_grad()

            if args.apex and args.use_npu:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            batch_errs, batch_tokens = model.module.compute_wer(index.numpy(), input_sizes.numpy(), targets.numpy(),
                                                                target_sizes.numpy())
            total_errs += batch_errs
            total_tokens += batch_tokens
            batch_time += (time.time() - end)

        else:
            batch_errs, batch_tokens = model.module.compute_wer(index.numpy(), input_sizes.numpy(), targets.numpy(),
                                                                target_sizes.numpy())
            total_errs += batch_errs
            total_tokens += batch_tokens

        if is_training:
            if i <= 3:
                print('Epoch: [%d] [%d / %d], Time %.6fs, Data %.6fs, FPS %.3f, total_loss = %.5f, total_wer = %.5f'
                      % (epoch_id, i + 1, steps_per_epoch, batch_time / (i + 1), data_time / (i + 1),
                         opts.batch_size * (i + 1) / batch_time, total_loss / (i + 1),
                         total_errs / total_tokens))
            else:
                print('Epoch: [%d] [%d / %d], Time %.6fs, Data %.6fs, FPS %.3f, total_loss = %.5f, total_wer = %.5f'
                      % (epoch_id, i + 1, steps_per_epoch, batch_time / (i - 3), data_time / (i - 3),
                         opts.batch_size * (i - 3) / batch_time, total_loss / (i + 1),
                         total_errs / total_tokens))
        end = time.time()

    average_loss = total_loss / (i + 1)
    training = "Train" if is_training else "Valid"
    print("Epoch %d %s done, total_loss: %.4f, total_wer: %.4f" % (
    epoch_id, training, average_loss, total_errs / total_tokens))
    return 1 - total_errs / total_tokens, average_loss


class Config(object):
    batch_size = 4
    dropout = 0.1


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')


def main(conf):
    args = parser.parse_args()
    opts = Config()
    for k, v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))
    print("===============main()=================")
    print(args)
    os.environ['MASTER_ADDR'] = args.addr  # '10.136.181.51'
    os.environ['MASTER_PORT'] = '29501'
    # if opts.gpu is not None:
    #     warnings.warn('You have chosen a specific GPU. This will completely '
    #                   'disable data parallelism.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.process_device_map = device_id_to_process_device_map(args.device_list)
    if args.device == 'npu':
        # npus_per_node = torch.npu.device_count()
        npus_per_node = len(args.process_device_map)
    else:
        npus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have npus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = npus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # The child process uses the environment variables of the parent process,
        # we have to set KERNEL_NAME_ID for every proc
        if args.device == 'npu':
            torch.multiprocessing.spawn(main_worker, nprocs=npus_per_node, args=(npus_per_node, args, opts))
        else:
            torch.multiprocessing.spawn(main_worker, nprocs=npus_per_node, args=(npus_per_node, argsm, opts))
    else:
        # Simply call main_worker function
        main_worker(args.device, npus_per_node, args, opts)


def main_worker(dev, npus_per_node, args, opts):
    device_id = args.process_device_map[dev]
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * npus_per_node + dev

        if args.device == 'npu':
            dist.init_process_group(backend=args.dist_backend,  # init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)

    loc = 'npu:{}'.format(device_id)
    if args.use_npu:
        torch.npu.set_device(loc)

    opts.batch_size = int(opts.batch_size / npus_per_node)
    args.workers = int((args.workers + npus_per_node - 1) / npus_per_node)

    print("[npu id:", device_id, "]", "===============main_worker()=================")
    print("[npu id:", device_id, "]", args)
    print("[npu id:", device_id, "]", "===============main_worker()=================")

    device = torch.device(loc) if args.use_npu else torch.device('cpu')

    sum_writer = SummaryWriter(opts.summary_path)
    if opts.seed is not None:
        seed_everything(opts.seed)

    # Data Loader
    vocab = Vocab(opts.vocab_file)
    train_dataset = SpeechDataset(vocab, opts.train_scp_path, opts.train_lab_path, opts)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = SpeechDataLoader(train_dataset, batch_size=opts.batch_size, shuffle=(train_sampler is None),
                                    num_workers=opts.num_workers, drop_last=True, pin_memory=True,
                                    sampler=train_sampler)

    dev_dataset = SpeechDataset(vocab, opts.valid_scp_path, opts.valid_lab_path, opts)
    if args.distributed:
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset)
    else:
        dev_sampler = None

    dev_loader = SpeechDataLoader(dev_dataset, batch_size=opts.batch_size, shuffle=(dev_sampler is None),
                                  num_workers=opts.num_workers,
                                  drop_last=True, pin_memory=True, sampler=dev_sampler)

    # Define Model
    rnn_type = supported_rnn[opts.rnn_type]
    rnn_param = {"rnn_input_size": opts.rnn_input_size, "rnn_hidden_size": opts.rnn_hidden_size,
                 "rnn_layers": opts.rnn_layers,
                 "rnn_type": rnn_type, "bidirectional": opts.bidirectional, "batch_norm": opts.batch_norm}

    num_class = vocab.n_words
    opts.output_class_dim = vocab.n_words
    drop_out = opts.drop_out
    add_cnn = opts.add_cnn

    cnn_param = {}
    channel = eval(opts.channel)
    kernel_size = eval(opts.kernel_size)
    stride = eval(opts.stride)
    padding = eval(opts.padding)
    pooling = eval(opts.pooling)
    activation_function = supported_activate[opts.activation_function]
    cnn_param['batch_norm'] = opts.batch_norm
    cnn_param['activate_function'] = activation_function
    cnn_param["layer"] = []
    for layer in range(opts.layers):
        layer_param = [channel[layer], kernel_size[layer], stride[layer], padding[layer]]
        if pooling is not None:
            layer_param.append(pooling[layer])
        else:
            layer_param.append(None)
        cnn_param["layer"].append(layer_param)

    seed = _gen_seeds(seed_shape)
    seed = torch.from_numpy(seed)
    seed = seed.to(device)
    model = CTC_Model(add_cnn=add_cnn, cnn_param=cnn_param, rnn_param=rnn_param, num_class=num_class, drop_out=drop_out,
                      seed=seed)

    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    print("Number of parameters %d" % num_params)
    for idx, m in enumerate(model.children()):
        print(idx, m)

    # Training
    init_lr = opts.init_lr
    num_epoches = opts.num_epoches
    end_adjust_acc = opts.end_adjust_acc
    decay = opts.lr_decay
    weight_decay = opts.weight_decay
    batch_size = opts.batch_size

    params = {'num_epoches': num_epoches, 'end_adjust_acc': end_adjust_acc, 'mel': opts.mel, 'seed': opts.seed,
              'decay': decay, 'learning_rate': init_lr, 'weight_decay': weight_decay, 'batch_size': batch_size,
              'feature_type': opts.feature_type, 'n_feats': opts.feature_dim}
    print(params)

    loss_fn = nn.CTCLoss(reduction='sum').to(device)

    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    model = model.to(device)

    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale, combine_grad=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], broadcast_buffers=False)

    count = 0
    learning_rate = init_lr
    loss_best = 1000
    loss_best_true = 1000
    adjust_rate_flag = False
    stop_train = False
    adjust_time = 0
    acc_best = 0
    start_time = time.time()
    loss_results = []
    dev_loss_results = []
    dev_cer_results = []

    while not stop_train:
        print(model.module.rnn_param)
        if count >= num_epoches:
            break
        count += 1

        if adjust_rate_flag:
            learning_rate *= decay
            adjust_rate_flag = False
            for param in optimizer.param_groups:
                param['lr'] *= decay

        print("Start training epoch: %d, learning_rate: %.5f" % (count, learning_rate))

        train_acc, loss = run_epoch(count, model, train_loader, loss_fn, device, args, opts, sum_writer,
                                    optimizer=optimizer, print_every=opts.verbose_step, is_training=True)
        loss_results.append(loss)
        acc, dev_loss = run_epoch(count, model, dev_loader, loss_fn, device, args, opts, sum_writer, optimizer=None,
                                  print_every=opts.verbose_step, is_training=False)
        print("loss on dev set is %.4f" % dev_loss)
        dev_loss_results.append(dev_loss)
        dev_cer_results.append(acc)

        # adjust learning rate by dev_loss
        if dev_loss < (loss_best - end_adjust_acc):
            loss_best = dev_loss
            loss_best_true = dev_loss
            adjust_rate_count = 0
            model_state = copy.deepcopy(model.state_dict())
            op_state = copy.deepcopy(optimizer.state_dict())
        elif (dev_loss < loss_best + end_adjust_acc):
            adjust_rate_count += 1
            if dev_loss < loss_best and dev_loss < loss_best_true:
                loss_best_true = dev_loss
                model_state = copy.deepcopy(model.state_dict())
                op_state = copy.deepcopy(optimizer.state_dict())
        else:
            adjust_rate_count = 10

        if acc > acc_best or count == 0:
            acc_best = acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_op_state = copy.deepcopy(optimizer.state_dict())

        print("adjust_rate_count:" + str(adjust_rate_count))
        print('adjust_time:' + str(adjust_time))

        if adjust_rate_count == 10:
            adjust_rate_flag = True
            adjust_time += 1
            adjust_rate_count = 0
            if loss_best > loss_best_true:
                loss_best = loss_best_true
            model.load_state_dict(model_state)
            optimizer.load_state_dict(op_state)

        if adjust_time == 20:
            stop_train = True

        time_used = (time.time() - start_time) / 60
        print("epoch %d done, cv acc is: %.4f, time_used: %.4f minutes" % (count, acc, time_used))

    print("End training, best dev loss is: %.4f, acc is: %.4f" % (loss_best, acc_best))
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_op_state)
    save_dir = os.path.join(opts.checkpoint_dir, opts.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_path = os.path.join(save_dir, 'ctc_best_model.pth')
    params['epoch'] = count

    if not args.multiprocessing_distributed or \
            (args.multiprocessing_distributed and args.rank % npus_per_node == 0):
        torch.save(CTC_Model.save_package(model.module, optimizer=optimizer, epoch=params, loss_results=loss_results,
                                          dev_loss_results=dev_loss_results, dev_cer_results=dev_cer_results),
                   best_path)


if __name__ == '__main__':
    args = parser.parse_args()
    try:
        config_path = args.conf
        conf = yaml.safe_load(open(config_path, 'r'))
    except:
        print("No input config or config file missing, please check.")
        sys.exit(1)
    main(conf)
