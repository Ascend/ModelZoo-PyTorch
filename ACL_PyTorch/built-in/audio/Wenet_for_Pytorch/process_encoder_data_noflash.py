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


from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import AudioDataset, CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
#from wenet.transformer.acl_init import decoder_model, device_id
import json
import os
import acl
from wenet.transformer.acl_net import Net
def dic2json(input_dict, json_path):
    json_str = json.dumps(input_dict)
    with open(json_path, 'a') as json_file:
        json_file.write(json_str)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--bin_path', type=str, default="./encoder_data_noflash", help='encoder bin images dir')
    parser.add_argument('--model_path', type=str, default="no_flash_encoder_revise.om", help='encoder bin images dir')
    parser.add_argument('--json_path', type=str, default="encoder_noflash_all.json", help='encoder bin images dir')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    args = parser.parse_args()
    print(args)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    #init acl
    ret = acl.init()
    device_id = 0
    #check_ret('acl.init', ret)
    ret = acl.rt.set_device(device_id)
    #check_ret('acl.rt.set_device', ret)
    context, ret = acl.rt.create_context(device_id)
    #check_ret('acl.rt.create_context', ret)

    decoder_output_data_shape = 42330000
    encoder_model_noflash = Net(
        model_path=args.model_path,
        output_data_shape=decoder_output_data_shape,
        device_id=device_id, )

    input_1 = np.random.random((1,200,80)).astype("float32")
    lenth = np.array([200])
    y, _ = encoder_model_noflash([input_1, lenth])

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    raw_wav = configs['raw_wav']
    # Init dataset and data loader
    # Init dataset and data loader
    test_collate_conf = copy.deepcopy(configs['collate_conf'])
    test_collate_conf['spec_aug'] = False
    test_collate_conf['spec_sub'] = False
    test_collate_conf['feature_dither'] = False
    test_collate_conf['speed_perturb'] = False
    if raw_wav:
        test_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0
    test_collate_func = CollateFunc(**test_collate_conf, raw_wav=raw_wav)
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_size'] = args.batch_size
    dataset_conf['batch_type'] = 'static'
    dataset_conf['sort'] = False
    test_dataset = AudioDataset(args.test_data,
                                **dataset_conf,
                                raw_wav=raw_wav)
    test_data_loader = DataLoader(test_dataset,
                                  collate_fn=test_collate_func,
                                  shuffle=False,
                                  batch_size=1,
                                  num_workers=0)

    # Init asr model from configs
    model = init_asr_model(configs)

    # Load dict
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    model.eval()
    if not os.path.exists(args.bin_path):
        os.mkdir(args.bin_path)
    #init acl
    if os.path.exists(args.json_path):
        os.remove(args.json_path)
    total_t = 0
    encoder_dic = {}
    import time
    for batch_idx, batch in enumerate(test_data_loader):
        print("batch_idx", batch_idx)
        keys, feats, target, feats_lengths, target_lengths = batch
        feats = feats.to(device)
        target = target.to(device)
        feats_lengths = feats_lengths.to(device)
        target_lengths = target_lengths.to(device)
        assert (feats.size(0) == 1)
        encoder_out, encoder_mask, exe_time = model.get_no_flash_encoder_out(
            encoder_model_noflash,
            batch_idx,
            feats,
            feats_lengths,
            args.beam_size,
            decoding_chunk_size=args.decoding_chunk_size,
            num_decoding_left_chunks=args.num_decoding_left_chunks,
            ctc_weight=args.ctc_weight,
            simulate_streaming=args.simulate_streaming,
            reverse_weight=args.reverse_weight)
        total_t += exe_time
        encoder_dic["encoder_out_"+ str(batch_idx)] = [encoder_out.shape[0], encoder_out.shape[1],encoder_out.shape[2]]
        encoder_dic["encoder_mask_"+ str(batch_idx)] = [encoder_mask.shape[0], encoder_mask.shape[1],encoder_mask.shape[2]]
        encoder_out.numpy().tofile(os.path.join(args.bin_path, "encoder_out_{}.bin".format(batch_idx)))
        encoder_mask.numpy().tofile(os.path.join(args.bin_path, "encoder_mask_{}.bin".format(batch_idx)))
    ave_t = total_t / (batch_idx + 1)
    dic_perf = {}
    dic_perf["t1"] = ave_t
    dic2json(dic_perf, "t1.json")
    dic2json(encoder_dic, args.json_path)
    del encoder_model_noflash
