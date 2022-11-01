# Copyright 2022 Huawei Technologies Co., Ltd
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

#!/usr/bin/python
#encoding=utf-8

import os
import sys
import copy
import time
import yaml
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.onnx
from collections import OrderedDict
import ssl
sys.path.append('./')
from models.model_ctc import *

supported_rnn = {'nn.LSTM':nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN':nn.RNN}
supported_activate = {'relu':nn.ReLU, 'tanh':nn.Tanh, 'sigmoid':nn.Sigmoid}

parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
parser.add_argument('--conf', default='conf/ctc_config.yaml', help='conf file with argument of LSTM and training')
parser.add_argument('--batchsize', default=1, help='batchszie for transfer onnx batch')

class Vocab(object):
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.word2index = {"blank": 0, "UNK": 1}
        self.index2word = {0: "blank", 1: "UNK"}
        self.word2count = {}
        self.n_words = 2
        self.read_lang()

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def read_lang(self):
        print("Reading vocabulary from {}".format(self.vocab_file))
        with open(self.vocab_file, 'r') as rf:
            line = rf.readline()
            while line:
                line = line.strip().split(' ')
                if len(line) > 1:
                    sen = ' '.join(line[1:])
                else:
                    sen = line[0]
                self.add_sentence(sen)
                line = rf.readline()
        print("Vocabulary size is {}".format(self.n_words))


def proc_nodes_module(checkpoint, AttrName):
    new_state_dict = OrderedDict()
    for k, v in checkpoint[AttrName].items():
        if(k[0:7] == "module."):
            name = k[7:]
        else:
            name = k[0:]

        new_state_dict[name]=v
    return new_state_dict

def run_epoch(epoch_id, model, data_iter, loss_fn, device, 
                opts, sum_writer, optimizer=None, print_every=20, is_training=True):
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
        data_time += (time.time() - end)

        global_step = epoch_id * steps_per_epoch + i
        inputs, input_sizes, targets, target_sizes, utt_list = data
        with torch.autograd.profiler.profile(record_shapes=True, use_cuda=True) as prof:
            inputs = inputs.to(device)
            input_sizes = input_sizes.to(device)
            targets = targets.to(device)
            target_sizes = target_sizes.to(device)
            out = model(inputs)
            out_len, batch_size, _ = out.size()
            input_sizes = (input_sizes * out_len).long()
            loss = loss_fn(out, targets, input_sizes, target_sizes)
            loss /= batch_size
            cur_loss += loss.item()
            total_loss += loss.item()
            prob, index = torch.max(out, dim=-1)
            batch_errs, batch_tokens = model.compute_wer(index.transpose(0, 1).cpu().numpy(), 
		input_sizes.cpu().numpy(), targets.cpu().numpy(), target_sizes.cpu().numpy())
            total_errs += batch_errs
            total_tokens += batch_tokens

            if is_training:
                optimizer.zero_grad()
                if opts.opt_level and opts.use_gpu:
                   with amp.scale_loss(loss, optimizer) as scaled_loss:
                       scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                sum_writer.add_scalar('Accuary/train/total_loss', total_loss / (i+1), global_step)
                sum_writer.add_scalar('Accuary/train/total_wer', total_errs / total_tokens, global_step)
        prof.export_chrome_trace('prof/'+str(i) + "_cuda_lstm.prof")
        batch_time += (time.time() - end)
        if is_training:
            print('Epoch: [%d] [%d / %d], Time %.6f Data %.6f s, total_loss = %.5f s, total_wer = %.5f' % (epoch_id,
                                    i+1, steps_per_epoch, batch_time / (i+1), data_time / (i+1), total_loss / (i+1), 
                                    total_errs / total_tokens ))
        end = time.time()


    average_loss = total_loss / (i+1)
    training = "Train" if is_training else "Valid"
    return 1-total_errs / total_tokens, average_loss

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

def main(conf, batchsize):
    checkpoint = torch.load("./checkpoint/ctc_fbank_cnn/ctc_best_model.pth", map_location='cpu')
    checkpoint['state_dict'] = proc_nodes_module(checkpoint, 'state_dict')
    opts = Config()
    for k, v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))

    device = torch.device('cpu')
    sum_writer = SummaryWriter(opts.summary_path)

    if opts.seed is not None:
        seed_everything(opts.seed)

    #Data Loader
    vocab = Vocab(opts.vocab_file)
    #Define Model
    rnn_type = supported_rnn[opts.rnn_type]
    rnn_param = {"rnn_input_size":opts.rnn_input_size, 
                    "rnn_hidden_size":opts.rnn_hidden_size, "rnn_layers":opts.rnn_layers,
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
    model = model.to('cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    input_names = ["actual_input_1"]
    output_names = ["output1"]
    batch_size = int(batchsize)
    dummy_input = torch.randn(batch_size, 390, 243, device='cpu')
    dynamic_axes = {'actual_input_1': {0: '-1'}, 'output1': {1: '-1'}}
    output_file = "lstm_ctc_{}batch.onnx".format(str(batch_size))
    torch.onnx.export(model, dummy_input, output_file, input_names = input_names, 
                        output_names = output_names, opset_version=11)
if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    args = parser.parse_args()
    batchsize = args.batchsize
    try:
        config_path = args.conf
        conf = yaml.safe_load(open(config_path, 'r'))
    except:
        print("No input config or config file missing, please check.")
        sys.exit(1)
    main(conf, batchsize)
