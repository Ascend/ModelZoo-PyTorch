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
import copy
import torch
import torch.nn.functional as F
import kaldi_io
import json
from data import build_LFR_features
from utils import (get_attn_pad_mask, get_non_pad_mask,
                   add_results_to_json, process_dict, get_subsequent_mask, pad_list)
import time
import argparse
import acl
from acl_net import Net


parser = argparse.ArgumentParser("Speech-Transformer-ominf")
parser.add_argument('--dict_path', type=str,
                    default='data/lang_1char/train_chars.txt')
parser.add_argument('--recog_json', type=str,
                    default='dump/test/deltafalse/data.json')
parser.add_argument('--encoder_path', type=str, default='./encoder.om')
parser.add_argument('--decoder_path', type=str, default='./decoder.om')
parser.add_argument('--tgt_word_prj_path', type=str,
                    default='./tgt_word_prj.om')
parser.add_argument('--result_label', type=str,
                    default='./om_decode/data.json')


class SpeechTransformer(object):

    def __init__(self, device_id, encoder_path, decoder_path, tgt_word_prj_path):
        ret = acl.init()
        assert ret == 0
        ret = acl.rt.set_device(device_id)
        assert ret == 0
        context, ret = acl.rt.create_context(device_id)
        assert ret == 0
        self.encoder_context = Net(context, device_id=device_id,
                                   model_path=encoder_path, first=True)
        self.decoder_context = Net(context, device_id=device_id,
                                   model_path=decoder_path, first=False)
        self.tgt_word_prj_context = Net(
            context, device_id=device_id, model_path=tgt_word_prj_path, first=False)

        self.beam = 5
        self.nbest = 1
        self.maxlen = 100

    def __call__(self, padded_input):
        return self.forward(padded_input)

    def forward(self, padded_input):

        input_lengths = torch.tensor(
            [padded_input.size(0)], dtype=torch.int).contiguous()

        padded_input = padded_input.unsqueeze(0)
        padded_input = pad_list(padded_input, 0, max_len=512)
        non_pad_mask = get_non_pad_mask(
            padded_input, input_lengths=input_lengths).contiguous().numpy()
        length = padded_input.size(1)
        slf_attn_mask = get_attn_pad_mask(
            padded_input, input_lengths, length).contiguous().numpy()
        padded_input = padded_input.numpy()

        encoder_outputs = self.encoder_context(
            (padded_input, non_pad_mask, slf_attn_mask))[0][0]

        # prepare sos
        ys = torch.zeros([1, 128]).long()
        ys[0, 0] = 1

        # yseq: 1xT
        hyp = {'score': 0.0, 'yseq': ys}
        hyps = [hyp]
        ended_hyps = []
        for i in range(self.maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                ys = hyp['yseq']  # 1 x i
                # -- Prepare masks
                non_pad_mask = torch.ones_like(
                    ys).float().unsqueeze(-1).numpy()
                slf_attn_mask = get_subsequent_mask(ys).numpy()
                ys = ys.numpy()

                dec_output = self.decoder_context(
                    (ys, encoder_outputs, non_pad_mask, slf_attn_mask))[0]
                seq_logit = torch.tensor(
                    self.tgt_word_prj_context(dec_output[:, i])[0])

                local_scores = F.log_softmax(seq_logit, dim=1)
                # topk scores
                local_best_scores, local_best_ids = torch.topk(
                    local_scores, self.beam, dim=1)

                for j in range(self.beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = copy.deepcopy(hyp['yseq'])
                    new_hyp['yseq'][:, i + 1] = int(local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(hyps_best_kept,
                                        key=lambda x: x['score'],
                                        reverse=True)[:self.beam]
                # end for hyp in hyps
            hyps = hyps_best_kept

            # add eos in the final loop to avoid that there are no ended hyps
            if i == self.maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'][:, self.maxlen] = 2

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][0, i + 1] == 2:
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)
            hyps = remained_hyps
            if len(hyps) == 0:
                print('no hypothesis. Finish decoding.')
                break

            # end for i in range(maxlen)
        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
            :min(len(ended_hyps), self.nbest)]
        return nbest_hyps


def main():
    args = parser.parse_args()
    char_list, _, _ = process_dict(args.dict_path)

    model = SpeechTransformer(1, args.encoder_path,
                              args.decoder_path, args.tgt_word_prj_path)
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}
    t = time.time()
    for idx, name in enumerate(js.keys(), 1):
        padded_input = kaldi_io.read_mat(js[name]['input'][0]['feat'])  # TxD
        padded_input = build_LFR_features(padded_input, 4, 3)
        padded_input = torch.from_numpy(padded_input).float()
        nbest_hyps = model(padded_input)
        index = torch.where(nbest_hyps[0]['yseq'][0] == 2)[0]
        nbest_hyps[0]['yseq'] = nbest_hyps[0]['yseq'][:,
                                                      :index + 1][0].cpu().numpy().tolist()
        new_js[name] = add_results_to_json(js[name], nbest_hyps, char_list)
        print('FPS: {:2f}'.format(idx / (time.time() - t)))

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4,
                           sort_keys=True).encode('utf_8'))


if __name__ == '__main__':
    main()
