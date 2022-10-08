#!/usr/bin/python
#encoding=utf-8
#
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
#

import os
import sys
import copy
import time 
import yaml
import argparse
import numpy as np
import apex
from apex import amp
import torch
import torch.nn as nn
import torch.npu
if torch.__version__ >= "1.8":
    import torch_npu

sys.path.append('./')
from models.model_ctc import *
#from warpctc_pytorch import CTCLoss # use built-in nn.CTCLoss
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader

supported_rnn = {'nn.LSTM':nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN':nn.RNN}
supported_activate = {'relu':nn.ReLU, 'tanh':nn.Tanh, 'sigmoid':nn.Sigmoid}

parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
parser.add_argument('--conf', default='conf/ctc_config.yaml' , help='conf file with argument of LSTM and training')
parser.add_argument('--device_id', default='0', type=str, help='device id')
parser.add_argument('--apex', action='store_true', help='User apex for mixed precision training')
parser.add_argument('--loss_scale', default=128.0, type=float, help='loss scale using in amp, default -1 means dynamic')
parser.add_argument('--opt_level', default='O2', type=str, help='loss scale using in amp, default -1 means dynamic')

def run_epoch(epoch_id, model, data_iter, loss_fn, device, optimizer=None, print_every=20, is_training=True):
    if is_training:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    total_tokens = 0
    total_errs = 0
    cur_loss = 0

    for i, data in enumerate(data_iter):
        start_time = time.time()
        inputs, input_sizes, targets, target_sizes, utt_list = data
        inputs = inputs.to(device,non_blocking=True)
        input_sizes = input_sizes.to(device,non_blocking=True)
        targets = targets.to(device,non_blocking=True)
        target_sizes = target_sizes.to(device,non_blocking=True)
       
        out = model(inputs)
        out_len, batch_size, _ = out.size()
        input_sizes = (input_sizes * out_len).long()
        loss = loss_fn(out, targets, input_sizes, target_sizes)
        loss /= batch_size
        cur_loss += loss.item()
        total_loss += loss.item()
        prob, index = torch.max(out, dim=-1)
        batch_errs, batch_tokens = model.compute_wer(index.transpose(0,1).cpu().numpy(), input_sizes.cpu().numpy(), targets.cpu().numpy(), target_sizes.cpu().numpy())
        total_errs += batch_errs
        total_tokens += batch_tokens

        if is_training:    
            optimizer.zero_grad()
            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 400)
            optimizer.step()
            step_time = time.time() - start_time

        if (i + 1) % print_every == 0 and is_training:
            print('Epoch = %d, step = %d, time = %.4f, cur_loss = %.4f, total_loss = %.4f, total_wer = %.4f' % (epoch_id, 
                                    i+1, step_time, cur_loss / print_every, total_loss / (i+1), total_errs / total_tokens ))
            cur_loss = 0
        
    average_loss = total_loss / (i+1)
    training = "Train" if is_training else "Valid"
    print("Epoch %d %s done, total_loss: %.4f, total_wer: %.4f" % (epoch_id, training, average_loss, total_errs / total_tokens))
    return 1-total_errs / total_tokens, average_loss

class Config(object):
    batch_size = 4
    dropout = 0.1

def main(args,conf):
    opts = Config()
    for k, v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))

    device = torch.device('npu:' + args.device_id) if opts.use_gpu else torch.device('cpu')
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    if opts.use_gpu:
        torch.npu.set_device(device)
        torch.npu.manual_seed(opts.seed)
    
    #Data Loader
    vocab = Vocab(opts.vocab_file)
    train_dataset = SpeechDataset(vocab, opts.train_scp_path, opts.train_lab_path, opts)
    dev_dataset = SpeechDataset(vocab, opts.valid_scp_path, opts.valid_lab_path, opts)
    train_loader = SpeechDataLoader(train_dataset, batch_size=opts.batch_size, shuffle=opts.shuffle_train, num_workers=opts.num_workers, drop_last=True, pin_memory=True)
    dev_loader = SpeechDataLoader(dev_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, drop_last=True, pin_memory=True)

    #Define Model
    rnn_type = supported_rnn[opts.rnn_type]
    rnn_param = {"rnn_input_size":opts.rnn_input_size, "rnn_hidden_size":opts.rnn_hidden_size, "rnn_layers":opts.rnn_layers, 
                    "rnn_type":rnn_type, "bidirectional":opts.bidirectional, "batch_norm":opts.batch_norm}
    
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

    model = CTC_Model(add_cnn=add_cnn, cnn_param=cnn_param, rnn_param=rnn_param, num_class=num_class, drop_out=drop_out)
    model = model.to(device)
    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    print("Number of parameters %d" % num_params)
    for idx, m in enumerate(model.children()):
        print(idx, m)
    
    #Training
    init_lr = opts.init_lr
    num_epoches = opts.num_epoches
    end_adjust_acc = opts.end_adjust_acc
    decay = opts.lr_decay
    weight_decay = opts.weight_decay
    batch_size = opts.batch_size
    
    params = { 'num_epoches':num_epoches, 'end_adjust_acc':end_adjust_acc, 'mel': opts.mel, 'seed':opts.seed,
                'decay':decay, 'learning_rate':init_lr, 'weight_decay':weight_decay, 'batch_size':batch_size,
                'feature_type':opts.feature_type, 'n_feats': opts.feature_dim }
    print(params)
    
    loss_fn = nn.CTCLoss(reduction='sum')
    optimizer = apex.optimizers.NpuFusedAdam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale,combine_grad=True)
    #visualization for training
    # from visdom import Visdom
    # viz = Visdom()
    # if add_cnn:
        # title = opts.feature_type + str(opts.feature_dim) + ' CNN_LSTM_CTC'
    # else:
        # title = opts.feature_type + str(opts.feature_dim) + ' LSTM_CTC'

    # viz_opts = [dict(title=title+" Loss", ylabel = 'Loss', xlabel = 'Epoch'),
                # dict(title=title+" Loss on Dev", ylabel = 'DEV Loss', xlabel = 'Epoch'),
                # dict(title=title+' CER on DEV', ylabel = 'DEV CER', xlabel = 'Epoch')]
    # viz_window = [None, None, None]
    
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
        if count >= num_epoches:
            break
        count += 1
        
        if adjust_rate_flag:
            learning_rate *= decay
            adjust_rate_flag = False
            for param in optimizer.param_groups:
                param['lr'] *= decay
        
        print("Start training epoch: %d, learning_rate: %.5f" % (count, learning_rate))
        
        train_acc, loss = run_epoch(count, model, train_loader, loss_fn, device, optimizer=optimizer, print_every=opts.verbose_step, is_training=True)
        loss_results.append(loss)
        acc, dev_loss = run_epoch(count, model, dev_loader, loss_fn, device, optimizer=None, print_every=opts.verbose_step, is_training=False)
        print("loss on dev set is %.4f" % dev_loss)
        dev_loss_results.append(dev_loss)
        dev_cer_results.append(acc)
        
        #adjust learning rate by dev_loss
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

        print("adjust_rate_count:"+str(adjust_rate_count))
        print('adjust_time:'+str(adjust_time))

        if adjust_rate_count == 10:
            adjust_rate_flag = True
            adjust_time += 1
            adjust_rate_count = 0
            if loss_best > loss_best_true:
                loss_best = loss_best_true
            model.load_state_dict(model_state)
            optimizer.load_state_dict(op_state)

        if adjust_time == 8:
            stop_train = True
        
        time_used = (time.time() - start_time) / 60
        print("epoch %d done, cv acc is: %.4f, time_used: %.4f minutes" % (count, acc, time_used))
        
        # x_axis = range(count)
        # y_axis = [loss_results[0:count], dev_loss_results[0:count], dev_cer_results[0:count]]
        # for x in range(len(viz_window)):
            # if viz_window[x] is None:
                # viz_window[x] = viz.line(X = np.array(x_axis), Y = np.array(y_axis[x]), opts = viz_opts[x],)
            # else:
                # viz.line(X = np.array(x_axis), Y = np.array(y_axis[x]), win = viz_window[x], update = 'replace',)
        
    print("End training, best dev loss is: %.4f, acc is: %.4f" % (loss_best, acc_best))
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_op_state)
    save_dir = os.path.join(opts.checkpoint_dir, opts.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_path = os.path.join(save_dir, 'ctc_best_model.pth')
    params['epoch']=count

    torch.save(CTC_Model.save_package(model, optimizer=optimizer, epoch=params, loss_results=loss_results, dev_loss_results=dev_loss_results, dev_cer_results=dev_cer_results), best_path)

if __name__ == '__main__':
    args = parser.parse_args()
    try:
        config_path = args.conf
        conf = yaml.safe_load(open(config_path, 'r'))
    except:
        print("No input config or config file missing, please check.")
        sys.exit(1)
    main(args,conf)
