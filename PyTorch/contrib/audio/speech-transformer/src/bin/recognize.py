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
import argparse
import json
import time

import torch
from apex import amp
import kaldi_io
from sp_transformer import Transformer
from utils import add_results_to_json, process_dict, pad_list
from data import build_LFR_features

parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Decoding.")
# data
parser.add_argument('--recog-json', type=str, required=True,
                    help='Filename of recognition data (json)')
parser.add_argument('--dict', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')
parser.add_argument('--result-label', type=str, required=True,
                    help='Filename of result label data (json)')
# model
parser.add_argument('--model-path', type=str, required=True,
                    help='Path to model file created by training')

def recognize(args):
    model, LFR_m, LFR_n = Transformer.load_model(args.model_path)
    model.eval()
    model.npu()
    char_list, sos_id, eos_id = process_dict(args.dict)
    assert model.decoder.sos_id == sos_id and model.decoder.eos_id == eos_id

    model= amp.initialize(model, opt_level="O2", loss_scale=128.0, combine_grad=True)
    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    # decode each utterance
    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            print('(%d/%d) decoding %s' %
                  (idx, len(js.keys()), name), flush=True)
            input = kaldi_io.read_mat(js[name]['input'][0]['feat'])  # TxD
            input = build_LFR_features(input, LFR_m, LFR_n)
            # input.shape torch.Size([141, 320])
            input = torch.from_numpy(input).float()
            input_length = torch.tensor([input.size(0)], dtype=torch.int)
            input = input.unsqueeze(0)
            input = pad_list(input, 0, max_len = 512)
            input = input.npu()
            input_length = input_length.npu()
            nbest_hyps = model(input, input_length)
            index = torch.where(nbest_hyps[0]['yseq'][0] == 2)[0]
            nbest_hyps[0]['yseq'] = nbest_hyps[0]['yseq'][:, :index + 1][0].cpu().numpy().tolist()
            new_js[name] = add_results_to_json(js[name], nbest_hyps, char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4,
                           sort_keys=True).encode('utf_8'))


if __name__ == "__main__":
    args = parser.parse_args()
    print(args, flush=True)
    recognize(args)
