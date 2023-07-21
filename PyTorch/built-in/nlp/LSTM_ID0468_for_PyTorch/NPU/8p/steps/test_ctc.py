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

import os
import time
import sys
import torch
if torch.__version__ >= "1.8":
    import torch_npu
import yaml
import argparse
import torch.nn as nn

sys.path.append('./')
from models.model_ctc import *
from utils.ctcDecoder import GreedyDecoder, BeamDecoder
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader
from steps.train_ctc import Config

parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='conf file for training')
parser.add_argument('--use_npu', default=True, type=str, help='use npu to train the model')
parser.add_argument('--device_id', default='0', type=str, help='device id')


def test():
    args = parser.parse_args()
    try:
        conf = yaml.safe_load(open(args.conf, 'r'))
    except:
        print("Config file not exist!")
        sys.exit(1)

    opts = Config()
    for k, v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))

    use_npu = args.use_npu
    device = torch.device('npu:' + str(args.device_id)) if use_npu else torch.device('cpu')

    model_path = os.path.join(opts.checkpoint_dir, opts.exp_name, 'ctc_best_model.pth')
    package = torch.load(model_path)

    rnn_param = package["rnn_param"]
    add_cnn = package["add_cnn"]
    cnn_param = package["cnn_param"]
    num_class = package["num_class"]
    feature_type = package['epoch']['feature_type']
    n_feats = package['epoch']['n_feats']
    drop_out = package['_drop_out']
    mel = opts.mel

    beam_width = opts.beam_width
    lm_alpha = opts.lm_alpha
    decoder_type = opts.decode_type
    vocab_file = opts.vocab_file

    vocab = Vocab(vocab_file)
    test_dataset = SpeechDataset(vocab, opts.test_scp_path, opts.test_lab_path, opts)
    test_loader = SpeechDataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False,
                                   num_workers=opts.num_workers, pin_memory=False)

    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, cnn_param=cnn_param, num_class=num_class, drop_out=drop_out)
    model.to(device)
    model.load_state_dict(package['state_dict'])
    model.eval()

    if decoder_type == 'Greedy':
        decoder = GreedyDecoder(vocab.index2word, space_idx=-1, blank_index=0)
    else:
        decoder = BeamDecoder(vocab.index2word, beam_width=beam_width, blank_index=0, space_idx=-1,
                              lm_path=opts.lm_path, lm_alpha=opts.lm_alpha)

    total_wer = 0
    total_cer = 0
    start = time.time()
    with torch.no_grad():
        for data in test_loader:
            inputs, input_sizes, targets, target_sizes, utt_list = data
            inputs = inputs.to(device)

            probs = model(inputs)

            max_length = probs.size(0)
            input_sizes = (input_sizes * max_length).long()

            probs = probs.cpu()
            decoded = decoder.decode(probs, input_sizes.numpy().tolist())

            targets, target_sizes = targets.numpy(), target_sizes.numpy()
            labels = []
            for i in range(len(targets)):
                label = [vocab.index2word[num] for num in targets[i][:target_sizes[i]]]
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
    CER = (float(total_cer) / decoder.num_char) * 100
    WER = (float(total_wer) / decoder.num_word) * 100
    print("Character error rate on test set: %.4f" % CER)
    print("Word error rate on test set: %.4f" % WER)
    end = time.time()
    time_used = (end - start) / 60.0
    print("time used for decode %d sentences: %.4f minutes." % (len(test_dataset), time_used))


if __name__ == "__main__":
    test()
