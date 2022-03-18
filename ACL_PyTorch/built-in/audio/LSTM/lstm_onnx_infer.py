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

import os
import time
import sys
import torch
import yaml
import argparse
import onnxruntime
import torch.nn as nn
import numpy as np
from glob import glob
from tqdm import tqdm

sys.path.append('./')
from models.model_ctc import *
from utils.ctcDecoder import GreedyDecoder, BeamDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='conf file for training')

parser.add_argument('--model_path', required=True)
parser.add_argument('--bin_file_path', required=True)
parser.add_argument('--pred_res_save_path', required=True)
parser.add_argument('--batchsize', required=True, help='batchsize for onnx infering')

class Config(object):
    batch_size = 4
    dropout = 0.1

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

def lstm_onnx_infer():
    args = parser.parse_args()

    model_path = args.model_path
    bin_file_path = args.bin_file_path
    pred_res_save_path = args.pred_res_save_path

    try:
        conf = yaml.safe_load(open(args.conf,'r'))
    except:
        print("Config file not exist!")
        sys.exit(1)

    opts = Config()
    for k,v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))

    beam_width = opts.beam_width
    lm_alpha = opts.lm_alpha
    decoder_type =  opts.decode_type
    vocab_file = opts.vocab_file
    batchsize = int(args.batchsize)

    vocab = Vocab(vocab_file)

    # 读取数据目录
    bin_file_list = glob(os.path.join(bin_file_path, '*.bin'))
    bin_file_num = len(bin_file_list)

    # 创建目录
    pardir = os.path.dirname(pred_res_save_path)
    if not os.path.exists(pardir):
        os.makedirs(pardir)

    # 推理
    print('[INFO] Infer on dataset ...')
    transcription_list = []
    total_infer_time = 0
    total_infer_num = 0

    with open(pred_res_save_path, 'wt', encoding='utf-8') as f_pred:
        onnx_run_sess = onnxruntime.InferenceSession(model_path)
        for i in tqdm(range(bin_file_num)):
            # 数据预处理
            input = np.fromfile(os.path.join(bin_file_path, 'inputs_' + str(i) + '.bin'), dtype=np.float32)
            input = input.reshape(batchsize, 390, 243)

            # 推理
            start_time = time.perf_counter_ns()
            output = onnx_run_sess.run(None, {"actual_input_1":input})
            end_time = time.perf_counter_ns()
            total_infer_time += end_time - start_time
            total_infer_num += 1

        #推理时间
        print('[INFO] Time:')
        msg = 'total infer num: ' + str(total_infer_num) + '\n' + \
              'total infer time(ms): ' + str(total_infer_time / 1000 / 1000) + '\n' + \
              'average infer time(ms): ' + str(total_infer_time / total_infer_num / 1000 / 1000) + '\n'
        print(msg)
        with open(os.path.join(pardir, 'infer_time.txt'), 'wt', encoding='utf-8') as f_infer_time:
            f_infer_time.write(msg)


if __name__ == '__main__':
    '''
    Using Example:

    python onnx_local_infer.py \
    --conf=./conf/ctc_config.yaml \
    --model_path=./lstm_onnx/lstm_ctc_npu_16batch1_67.onnx \
    --bin_file_path=--bin_file_path=./lstm_bin/ \
    --pred_res_save_path=./lstm_onnx_infer \
    --batchsize=16
    '''
    lstm_onnx_infer()
