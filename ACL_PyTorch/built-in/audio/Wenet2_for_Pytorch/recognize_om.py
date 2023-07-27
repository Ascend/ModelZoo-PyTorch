# BSD 3-Clause License
#
# All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
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

"""
This script is for testing exported ascend encoder and decoder from
export_onnx_npu.py. The exported ascend models only support batch offline ASR inference.
It requires a python wrapped c++ ctc decoder.
Please install it from ctc decoder in github
"""
from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import time
import stat

import multiprocessing
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.common import IGNORE_ID
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.config import override_config
try:
    from swig_decoders import map_batch, \
        ctc_beam_search_decoder_batch, \
        TrieVector, PathTrie
except ImportError:
    print('Please install ctc decoders first')
    sys.exit(1)

from ais_bench.infer.interface import InferSession


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')

    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--encoder_om', required=True,
                        help='encoder om file')
    parser.add_argument('--decoder_om', required=True,
                        help='decoder om file')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--test_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='asr result file')
    parser.add_argument('--device_id',
                        type=int,
                        default=0,
                        help='npu device id')
    parser.add_argument('--mode',
                        choices=[
                            'ctc_greedy_search', 'ctc_prefix_beam_search',
                            'attention_rescoring'],
                        default='attention_rescoring',
                        help='decoding mode')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument('--fp16',
                        action='store_true',
                        help='whether to export fp16 model, default false')
    parser.add_argument('--static',
                        action='store_true',
                        help='whether to run static model')
    parser.add_argument('--num_process',
                        type=int,
                        default=2,
                        help='number of mutiprocesses')
    parser.add_argument('--encoder_gears',
                        type=str,
                        help='dynamic dims gears info for encoder, please input split by ","')
    parser.add_argument('--decoder_gears',
                        type=str,
                        help='dynamic dims gears info for encoder, please input split by ","')
    parser.add_argument('--output_size',
                        type=str,
                        help='only effect in dynamic shapes mode,\
                              outputs size info for encoder, please input split by ","')
    args_ = parser.parse_args()
    print(args_)
    return args_

def get_dict(dict_path):
    vocabulary_ = []
    char_dict_ = {}
    with open(dict_path, 'r') as fin_:
        for line in fin_:
            arr = line.strip().split()
            if len(arr) != 2:
                print('dict format is incorrect')
                exit(0)
            char_dict_[int(arr[1])] = arr[0]
            vocabulary_.append(arr[0])
    return vocabulary_, char_dict_

def init_om_session(device_id, encoder_path, decoder_path, mode):
    encoder_ort_session = InferSession(device_id, encoder_path)
    decoder_ort_session = None
    if mode == "attention_rescoring":
        decoder_ort_session = InferSession(device_id, decoder_path)
    return encoder_ort_session, decoder_ort_session

def adjust_test_conf(test_conf_, batch_size_):
    # adjust dataset parameters for om
    # reserved suitable memory
    test_conf_['filter_conf']['max_length'] = 102400
    test_conf_['filter_conf']['min_length'] = 0
    test_conf_['filter_conf']['token_max_length'] = 102400
    test_conf_['filter_conf']['token_min_length'] = 0
    test_conf_['filter_conf']['max_output_input_ratio'] = 102400
    test_conf_['filter_conf']['min_output_input_ratio'] = 0
    test_conf_['speed_perturb'] = False
    test_conf_['spec_aug'] = False
    test_conf_['shuffle'] = False
    test_conf_['sort'] = False
    test_conf_['fbank_conf']['dither'] = 0.0
    test_conf_['batch_conf']['batch_type'] = "static"
    test_conf_['batch_conf']['batch_size'] = batch_size_
    return test_conf_    

class AsrOmModel:
    def __init__(self, args_, reverse_weight_) -> None:
        self.encoder, self.decoder = init_om_session(args_.device_id, 
                                                     args_.encoder_om,
                                                     args_.decoder_om,
                                                     args_.mode)
        self.vocabulary, self.char_dict = get_dict(args_.dict)
        self.reverse_weight = reverse_weight_
        self.mul_shape = list(map(int, args.encoder_gears.split(',')))
        self.mul_shape_decoder = list(map(int, args.decoder_gears.split(',')))
        self.output_size = None
        if args.output_size:
            self.output_size = list(map(int, args.output_size.split(',')))
        self.args = args_
    
    def forward(self, data):
        nums_ = 0
        eos = sos = len(self.char_dict) - 1
        mode = "dymdims"
        if not self.args.static:
            mode = "dymshape"
        keys, feats, _, feats_lengths, _ = data
        feats, feats_lengths = feats.numpy(), feats_lengths.numpy()
        ort_outs = None
        if self.args.fp16:
            feats = feats.astype(np.float16)
        if self.args.static:
            pad_size = 0
            pad_batch = 0
            for n in self.mul_shape:
                if n > feats.shape[1]:
                    pad_size = n - feats.shape[1]
                    break
            if feats.shape[0] < self.args.batch_size:
                pad_batch = self.args.batch_size - feats.shape[0]
                feats_lengths = np.pad(feats_lengths, [(0, pad_batch)], 'constant')
            feats_pad = np.pad(feats, [(0, pad_batch), (0, pad_size), (0, 0)], 'constant')
            ort_outs = self.encoder.infer(
                [feats_pad, feats_lengths], mode)
        else:
            ort_outs = self.encoder.infer([feats, feats_lengths], mode, self.output_size)
        encoder_out, encoder_out_lens, _, \
            beam_log_probs, beam_log_probs_idx = ort_outs
        beam_size = beam_log_probs.shape[-1]
        batch_size = beam_log_probs.shape[0]
        num_processes = min(multiprocessing.cpu_count(), batch_size)
        if self.args.mode == 'ctc_greedy_search':
            if beam_size != 1:
                log_probs_idx = beam_log_probs_idx[:, :, 0]
            batch_sents = []
            for idx_, seq in enumerate(log_probs_idx):
                batch_sents.append(seq[0:encoder_out_lens[idx_]].tolist())
            hyps = map_batch(batch_sents, self.vocabulary, num_processes,
                                True, 0)
        elif self.args.mode in ('ctc_prefix_beam_search', "attention_rescoring"):
            batch_log_probs_seq_list = beam_log_probs.tolist()
            batch_log_probs_idx_list = beam_log_probs_idx.tolist()
            batch_len_list = encoder_out_lens.tolist()
            batch_log_probs_seq = []
            batch_log_probs_ids = []
            batch_start = []  # only effective in streaming deployment
            batch_root = TrieVector()
            root_dict = {}
            for i in range(len(batch_len_list)):
                num_sent = batch_len_list[i]
                batch_log_probs_seq.append(
                    batch_log_probs_seq_list[i][0:num_sent])
                batch_log_probs_ids.append(
                    batch_log_probs_idx_list[i][0:num_sent])
                root_dict[i] = PathTrie()
                batch_root.append(root_dict[i])
                batch_start.append(True)
            score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq,
                                                        batch_log_probs_ids,
                                                        batch_root,
                                                        batch_start,
                                                        beam_size,
                                                        num_processes,
                                                        0, -2, 0.99999)
            if self.args.mode == 'ctc_prefix_beam_search':
                hyps = []
                for cand_hyps in score_hyps:
                    hyps.append(cand_hyps[0][1])
                hyps = map_batch(hyps, self.vocabulary, num_processes, False, 0)
        if self.args.mode == 'attention_rescoring':
            ctc_score, all_hyps = [], []
            max_len = 0
            for hyps in score_hyps:
                cur_len = len(hyps)
                if len(hyps) < beam_size:
                    hyps += (beam_size - cur_len) * ((-float("INF"), (0,)),)
                cur_ctc_score = []
                for hyp in hyps:
                    cur_ctc_score.append(hyp[0])
                    all_hyps.append(list(hyp[1]))
                    if len(hyp[1]) > max_len:
                        max_len = len(hyp[1])
                ctc_score.append(cur_ctc_score)
            if self.args.fp16:
                ctc_score = np.array(ctc_score, dtype=np.float16)
            else:
                ctc_score = np.array(ctc_score, dtype=np.float32)
            hyps_pad_sos_eos = np.ones(
                (batch_size, beam_size, max_len + 2), dtype=np.int64) * IGNORE_ID
            r_hyps_pad_sos_eos = np.ones(
                (batch_size, beam_size, max_len + 2), dtype=np.int64) * IGNORE_ID
            hyps_lens_sos = np.ones(
                (batch_size, beam_size), dtype=np.int32)
            k = 0
            for i in range(batch_size):
                for j in range(beam_size):
                    cand = all_hyps[k]
                    l = len(cand) + 2
                    hyps_pad_sos_eos[i][j][0:l] = [sos] + cand + [eos]
                    r_hyps_pad_sos_eos[i][j][0:l] = [
                        sos] + cand[::-1] + [eos]
                    hyps_lens_sos[i][j] = len(cand) + 1
                    k += 1
            best_index = None
            if not self.args.static:
                output_size = 100000
                if self.reverse_weight > 0:
                    best_index = self.decoder.infer(
                        [encoder_out, encoder_out_lens, hyps_pad_sos_eos, hyps_lens_sos, 
                        r_hyps_pad_sos_eos, ctc_score], mode, output_size)
                else:
                    best_index = self.decoder.infer(
                        [encoder_out, encoder_out_lens, hyps_pad_sos_eos, hyps_lens_sos, 
                         ctc_score], mode, output_size)
            else:
                pad_size = 0
                for n in self.mul_shape_decoder:
                    if n > encoder_out.shape[1]:
                        pad_size = n - encoder_out.shape[1]
                        break
                encoder_out = np.pad(encoder_out, ((0, 0), (0, pad_size), (0, 0)), 'constant')
                if self.reverse_weight > 0:
                    best_index = self.decoder.infer(
                    [encoder_out, encoder_out_lens, hyps_pad_sos_eos, hyps_lens_sos, r_hyps_pad_sos_eos, ctc_score],
                    mode)
                else:
                    best_index = self.decoder.infer(
                        [encoder_out, encoder_out_lens, hyps_pad_sos_eos, hyps_lens_sos, ctc_score], mode)
            best_index = best_index[0]
            best_sents = []
            k = 0
            for idx_ in best_index:
                cur_best_sent = all_hyps[k: k + beam_size][idx_]
                best_sents.append(cur_best_sent)
                k += beam_size
            hyps = map_batch(best_sents, self.vocabulary, num_processes)

        for i, key in enumerate(keys):
            nums_ += 1
            content = hyps[i]
            logger.info('{} {}'.format(key, content))
        return nums_

def infer_process(idx_):
    batches = packed_data[idx_]
    init_start = time.time()
    model = AsrOmModel(args, reverse_weight)
    init_end = time.time()
    sync_num.append(1)
    while (len(sync_num) != args.num_process):
        # sync mutiple processes
        time.sleep(0.05)
    
    nums_ = 0
    for data in batches:
        num = model.forward(data)
        nums_ += num
    sync_num.pop()
    return nums_, init_end - init_start

if __name__ == '__main__':
    args = get_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(args.result_file, mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(level=logging.DEBUG)
    console_format = logging.Formatter('Recognize: %(message)s')
    console.setFormatter(console_format)
    logger.addHandler(handler)
    logger.addHandler(console)
    with open(args.config, 'r') as fin:
        configs = yaml.safe_load(fin)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    reverse_weight = configs["model_conf"].get("reverse_weight", 0.0)
    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf = adjust_test_conf(test_conf, args.batch_size)

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)
    manager = multiprocessing.Manager()
    num_process = args.num_process
    # packed data for mitiple processes
    packed_data = [[] for _ in range(num_process)]
    idx = 0
    for batch in test_data_loader:
        packed_data[idx].append(batch)
        idx = (idx + 1) % num_process

    sync_num = manager.list()
    data_cnt = 0
    init_times = 0
    total_time = 0
    start = time.time()

    with multiprocessing.Pool(num_process) as p:
        for nums, init_time in list(p.map(infer_process, range(num_process))):
            init_times = max(init_times, init_time)
            data_cnt += nums

    end = time.time()
    fps = float((data_cnt) / (end - start - init_times))
    fps_str = "fps: {}\n".format(fps)
    resstr = "total time: {}\n".format(end - start - init_times)
    print(fps_str)
    print(resstr)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL 
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(args.test_file, flags, modes), 'w') as f:
        f.write(fps_str)
        f.write(resstr)
