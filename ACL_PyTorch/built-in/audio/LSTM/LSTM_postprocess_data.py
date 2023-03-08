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

#!/usr/bin/python
#encoding=utf-8

import os
import time
import sys
import torch
import yaml
import argparse
import torch.nn as nn
import numpy as np

sys.path.append('./')
from models.model_ctc import *
from utils.ctcDecoder import GreedyDecoder, BeamDecoder
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='conf file for training')
parser.add_argument('--npu_path', help='infer file for postprocessing')
parser.add_argument('--batchsize', help='batchsize for postprocessing')

class Config(object):
    batch_size = 4
    dropout = 0.1

def test():
    args = parser.parse_args()
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

    vocab = Vocab(vocab_file)
    batchsize = int(args.batchsize)

    test_dataset = SpeechDataset(vocab, opts.valid_scp_path, opts.valid_lab_path, opts)
    test_loader = SpeechDataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=opts.num_workers, pin_memory=False)

    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(vocab.index2word, space_idx=-1, blank_index=0)
    else:
        decoder = BeamDecoder(vocab.index2word, beam_width=beam_width, blank_index=0, space_idx=-1, lm_path=opts.lm_path, lm_alpha=opts.lm_alpha)

    total_wer = 0
    total_cer = 0
    start = time.time()

    npu_path = args.npu_path
    test_num = 399 // batchsize
    with torch.no_grad():
        for i, data in zip(range(test_num), test_loader):
            inputs, input_sizes, targets, target_sizes, utt_list = data
            probs_1_np  = np.load('{}/inputs_{}_0.npy'.format(npu_path, i))
            probs_1 = torch.from_numpy(probs_1_np)

            max_length = probs_1.size(0)
            input_sizes = (input_sizes * max_length).long()

            decoded = decoder.decode(probs_1, input_sizes.numpy().tolist())
            targets, target_sizes = targets.numpy(), target_sizes.numpy()
            labels = []
            for i in range(len(targets)):
                label = [ vocab.index2word[num] for num in targets[i][:target_sizes[i]]]
                labels.append(' '.join(label))

            for x in range(len(targets)):
                print("origin : " + labels[x])
                print("decoded: " + decoded[x])
            cer = 0
            wer = 0
            for x in range(len(labels)):
                cer += decoder.cer(decoded[x], labels[x])
                wer += decoder.wer(decoded[x], labels[x])
                decoder.num_word += len(labels[x].split())
                decoder.num_char += len(labels[x])
            total_cer += cer
            total_wer += wer

    CER = (float(total_cer) / decoder.num_char)*100
    WER = (float(total_wer) / decoder.num_word)*100
    print("Character error rate on test set: %.4f" % CER)
    print("Word error rate on test set: %.4f" % WER)
    end = time.time()
    time_used = (end - start) / 60.0
    print("time used for decode %d sentences: %.4f minutes." % (len(test_dataset), time_used))

if __name__ == "__main__":
    test()
