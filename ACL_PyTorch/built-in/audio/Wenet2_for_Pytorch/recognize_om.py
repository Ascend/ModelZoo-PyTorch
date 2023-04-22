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
Please install it by following:
https://github.com/Slyne/ctc_decoder.git
"""
from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.common import IGNORE_ID
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.config import override_config

import onnxruntime as rt
import multiprocessing
import numpy as np
from pyacl.acl_infer import AclNet, init_acl, release_acl

try:
    from swig_decoders import map_batch, \
        ctc_beam_search_decoder_batch, \
        TrieVector, PathTrie
except ImportError:
    print('Please install ctc decoders first by refering to\n' +
          'https://github.com/Slyne/ctc_decoder.git')
    sys.exit(1)


def _pad_sequence(sequences, batch_first=False, padding_value=0, mul_shape=None, batch_size=1, input_batch_size=1):
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]

    max_size = sequences[0].size()
    trailing_dims = max_size[1:]

    max_len = max([s.size(0) for s in sequences])
    if mul_shape is not None:
        for in_shape in mul_shape:
            if max_len < in_shape:
                max_len = in_shape
                break
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    if batch_first and input_batch_size < batch_size:
        tmp = torch.zeros([batch_size - input_batch_size,
                          out_tensor.shape[1], out_tensor.shape[2]], dtype=torch.float32)
        out_tensor = torch.cat((out_tensor, tmp), 0)
    return out_tensor


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
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    reverse_weight = configs["model_conf"].get("reverse_weight", 0.0)
    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['fbank_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    init_acl(args.device_id)
    encoder_ort_session = AclNet(
        model_path=args.encoder_om, device_id=args.device_id, input_data_shape=[
            120000*args.batch_size, 1*args.batch_size], output_data_shape=[95744*args.batch_size, 
            1*args.batch_size, 1583142*args.batch_size, 3704*args.batch_size, 3680*args.batch_size])
    decoder_ort_session = None
    if args.mode == "attention_rescoring":
        decoder_ort_session = AclNet(model_path=args.decoder_om, device_id=args.device_id, input_data_shape=[
            args.batch_size*384*256, args.batch_size, args.batch_size*50*10, args.batch_size*10, args.batch_size*50*10,
            args.batch_size*10], output_data_shape=[1*args.batch_size])

    # Load dict
    vocabulary = []
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
            vocabulary.append(arr[0])
    eos = sos = len(char_dict) - 1
    mul_shape = [262, 326, 390, 454, 518, 582, 646,
                 710, 774, 838, 902, 966, 1028, 1284, 1478]
    mul_shape_decoder = [96, 144, 384]
    sumt1 = 0
    sumt2 = 0
    data_cnt = 0
    total_time = 0
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for _, batch in enumerate(test_data_loader):
            data_cnt = data_cnt + 1
            keys, feats, _, feats_lengths, _ = batch
            feats, feats_lengths = feats.numpy(), feats_lengths.numpy()
            ort_outs = None
            t1 = 0
            if args.fp16:
                feats = feats.astype(np.float16)
            start_time = time.time()
            if args.static:
                feats_pad = _pad_sequence([torch.from_numpy(
                    x) for x in feats], True, 0, mul_shape, args.batch_size, feats.shape[0])
                dims1 = {'dimCount': 4, 'name': '', 'dims': [
                    args.batch_size, feats_pad.shape[1], 80, args.batch_size]}
                feats_pad = feats_pad.numpy()
                ort_outs, t1 = encoder_ort_session(
                    [feats_pad, feats_lengths], dims=dims1)
            else:
                ort_outs, t1 = encoder_ort_session([feats, feats_lengths])
            end_time = time.time()
            total_time += (end_time - start_time)
            sumt1 = sumt1 + t1
            encoder_out, encoder_out_lens, ctc_log_probs, \
                beam_log_probs, beam_log_probs_idx = ort_outs
            beam_size = beam_log_probs.shape[-1]
            batch_size = beam_log_probs.shape[0]
            num_processes = min(multiprocessing.cpu_count(), batch_size)
            if args.mode == 'ctc_greedy_search':
                if beam_size != 1:
                    log_probs_idx = beam_log_probs_idx[:, :, 0]
                batch_sents = []
                for idx, seq in enumerate(log_probs_idx):
                    batch_sents.append(seq[0:encoder_out_lens[idx]].tolist())
                hyps = map_batch(batch_sents, vocabulary, num_processes,
                                 True, 0)
            elif args.mode in ('ctc_prefix_beam_search', "attention_rescoring"):
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
                if args.mode == 'ctc_prefix_beam_search':
                    hyps = []
                    for cand_hyps in score_hyps:
                        hyps.append(cand_hyps[0][1])
                    hyps = map_batch(hyps, vocabulary, num_processes, False, 0)
            if args.mode == 'attention_rescoring':
                ctc_score, all_hyps = [], []
                max_len = 0
                for hyps in score_hyps:
                    cur_len = len(hyps)
                    if len(hyps) < beam_size:
                        hyps += (beam_size - cur_len) * [(-float("INF"), (0,))]
                    cur_ctc_score = []
                    for hyp in hyps:
                        cur_ctc_score.append(hyp[0])
                        all_hyps.append(list(hyp[1]))
                        if len(hyp[1]) > max_len:
                            max_len = len(hyp[1])
                    ctc_score.append(cur_ctc_score)
                if args.fp16:
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
                t2 = 0
                T2 = hyps_pad_sos_eos.shape[2]
                dims2 = None
                if not args.static:
                    if reverse_weight > 0:
                        best_index, t2 = decoder_ort_session(
                            [encoder_out, encoder_out_lens, hyps_pad_sos_eos, hyps_lens_sos, 
                            r_hyps_pad_sos_eos, ctc_score])
                    else:
                        best_index, t2 = decoder_ort_session(
                            [encoder_out, encoder_out_lens, hyps_pad_sos_eos, hyps_lens_sos, ctc_score])
                else:
                    encoder_out_pad = _pad_sequence([torch.from_numpy(
                        x) for x in encoder_out], True, 0, mul_shape_decoder, args.batch_size, encoder_out.shape[0])
                    encoder_out = encoder_out_pad.numpy()
                    if reverse_weight > 0:
                        dims2 = {'dimCount': 14, 'name': '', 'dims': [args.batch_size, encoder_out.shape[1],
                                encoder_out.shape[2], args.batch_size, args.batch_size, 10, T2, args.batch_size, 
                                10, args.batch_size, 10, T2, args.batch_size, 10]}

                    else:
                        dims2 = {'dimCount': 11, 'name': '', 'dims': [args.batch_size, encoder_out.shape[1], 
                                encoder_out.shape[2], args.batch_size, args.batch_size, 10, T2, 
                                args.batch_size, 10, args.batch_size, 10]}

                    if reverse_weight > 0:
                        best_index, t2 = decoder_ort_session(
                        [encoder_out, encoder_out_lens, hyps_pad_sos_eos, hyps_lens_sos, r_hyps_pad_sos_eos, ctc_score],
                        dims=dims2)
                    else:
                        best_index, t2 = decoder_ort_session(
                            [encoder_out, encoder_out_lens, hyps_pad_sos_eos, hyps_lens_sos, ctc_score], dims=dims2)
                sumt2 = sumt1 + t2
                best_index = best_index[0]
                best_sents = []
                k = 0
                for idx in best_index:
                    cur_best_sent = all_hyps[k: k + beam_size][idx]
                    best_sents.append(cur_best_sent)
                    k += beam_size
                hyps = map_batch(best_sents, vocabulary, num_processes)

            for i, key in enumerate(keys):
                content = hyps[i]
                logging.info('{} {}'.format(key, content))
                fout.write('{} {}\n'.format(key, content))
        fps = float(1000/((sumt1+sumt2)/(data_cnt*args.batch_size)))
        resstr = "total time: {}\n".format(total_time)
        with open(args.test_file, "a") as resfile:
            resfile.write(resstr)
    encoder_ort_session.release_model()
    if args.mode == "attention_rescoring":
        decoder_ort_session.release_model()
    release_acl(args.device_id)


if __name__ == '__main__':
    main()
