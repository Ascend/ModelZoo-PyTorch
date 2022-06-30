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
import numpy as np
import random
import torch
if torch.__version__ >= "1.8.1":
    import torch_npu
import torch.nn as nn
import torch.backends.cudnn as cudnn
import apex
from torch.utils.tensorboard import SummaryWriter
sys.path.append('./')
from models.model_ctc import *
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader
from apex import amp
import torch.npu
import torch.optim

supported_rnn = {'nn.LSTM': nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN': nn.RNN}
supported_activate = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}

parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
parser.add_argument('--conf', default='conf/ctc_config.yaml', help='conf file with argument of LSTM and training')
parser.add_argument('--use_npu', default=True, type=str, help='use npu to train the model')
parser.add_argument('--device_id', default='0', type=str, help='device id')
parser.add_argument('--apex', action='store_true',
                                       help='User apex for mixed precision training')
parser.add_argument('--loss_scale', default=128., type=float,
                    help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt_level', default='O2', type=str,
                    help='loss scale using in amp, default -1 means dynamic')

MAX = 2147483647
def _gen_seeds(shape):
    return np.random.uniform(1, MAX, size=shape).astype(np.float32)
seed_shape = (32 * 1024 * 12, )


def run_epoch(epoch_id, model, data_iter, loss_fn, device, opts, sum_writer, optimizer=None, print_every=20,
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
            #if args.opt_level and args.use_npu:
            if args.apex and args.use_npu:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            batch_errs, batch_tokens = model.compute_wer(index.numpy(), input_sizes.numpy(), targets.numpy(),
                                                         target_sizes.numpy())
            total_errs += batch_errs
            total_tokens += batch_tokens

            batch_time += (time.time() - end)
            # sum_writer.add_scalar('Accuary/train/total_loss', total_loss / (i+1), global_step)
            # sum_writer.add_scalar('Accuary/train/total_wer', total_errs / total_tokens, global_step)
        else:
            batch_errs, batch_tokens = model.compute_wer(index.numpy(), input_sizes.numpy(), targets.numpy(),
                                                         target_sizes.numpy())
            total_errs += batch_errs
            total_tokens += batch_tokens
            # sum_writer.add_scalar('Accuary/valid/total_loss', total_loss / (i+1), global_step)
            # sum_writer.add_scalar('Accuary/valid/total_wer', total_errs / total_tokens, global_step)

        if is_training:
            if i <= 3:
                print('Epoch: [%d] [%d / %d], Time %.6fs, Data %.6fs, FPS %.3f, total_loss = %.5f, total_wer = %.5f'
                      % (epoch_id, i + 1, steps_per_epoch, batch_time / (i + 1), data_time / (i + 1),
                         opts.batch_size * (i + 1) / batch_time , total_loss / (i + 1),
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
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args, conf):
    opts = Config()
    for k, v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))

    device = torch.device('npu:' + str(args.device_id)) if args.use_npu else torch.device('cpu')
    print("use device id is : ", device)
    if args.use_npu:
        torch.npu.set_device(device)

    sum_writer = SummaryWriter(opts.summary_path)
    if opts.seed is not None:
        seed_everything(opts.seed)

    # Data Loader
    vocab = Vocab(opts.vocab_file)
    train_dataset = SpeechDataset(vocab, opts.train_scp_path, opts.train_lab_path, opts)
    dev_dataset = SpeechDataset(vocab, opts.valid_scp_path, opts.valid_lab_path, opts)
    train_loader = SpeechDataLoader(train_dataset, batch_size=opts.batch_size, shuffle=opts.shuffle_train,
                                    num_workers=opts.num_workers, drop_last=True, pin_memory=True)
    dev_loader = SpeechDataLoader(dev_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,
                                  drop_last=True, pin_memory=True)

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
        print(model.rnn_param)
        if count >= num_epoches:
            break
        count += 1
        if adjust_rate_flag:
            learning_rate *= decay
            adjust_rate_flag = False
            for param in optimizer.param_groups:
                param['lr'] *= decay

        print("Start training epoch: %d, learning_rate: %.5f" % (count, learning_rate))

        train_acc, loss = run_epoch(count, model, train_loader, loss_fn, device, opts, sum_writer, optimizer=optimizer,
                                    print_every=opts.verbose_step, is_training=True)
        loss_results.append(loss)
        acc, dev_loss = run_epoch(count, model, dev_loader, loss_fn, device, opts, sum_writer, optimizer=None,
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

        if acc > acc_best:
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

    torch.save(CTC_Model.save_package(model, optimizer=optimizer, epoch=params, loss_results=loss_results,
                                      dev_loss_results=dev_loss_results, dev_cer_results=dev_cer_results), best_path)


if __name__ == '__main__':
    args = parser.parse_args()
    try:
        config_path = args.conf
        conf = yaml.safe_load(open(config_path, 'r'))
    except:
        print("No input config or config file missing, please check.")
        sys.exit(1)
    main(args, conf)
