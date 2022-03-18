import copy
import onnxruntime
import torch
import torch.nn.functional as F
import kaldi_io
import json
from data import build_LFR_features
from utils import (get_attn_pad_mask, get_non_pad_mask,
                   add_results_to_json, process_dict, get_subsequent_mask, pad_list)
import time
import argparse

parser = argparse.ArgumentParser("Speech-Transformer-ONNX-INF")
parser.add_argument('--dict_path', type=str,
                    default='data/lang_1char/train_chars.txt')
parser.add_argument('--recog_json', type=str,
                    default='dump/test/deltafalse/data.json')
parser.add_argument('--encoder_path', type=str, default='./encoder.onnx')
parser.add_argument('--decoder_path', type=str, default='./decoder.onnx')
parser.add_argument('--tgt_word_prj_path', type=str,
                    default='./tgt_word_prj.onnx')
parser.add_argument('--result_label', type=str,  default='./data.json')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_onnx_model(path):
    ort_session = onnxruntime.InferenceSession(path)
    return ort_session


def recognize_beam(encoder_outputs, decoder, tgt_word_prj):
    beam = 5
    nbest = 1
    maxlen = 100

    encoder_outputs = torch.tensor(encoder_outputs, dtype=torch.float)
    encoder_outputs = encoder_outputs.unsqueeze(0)
    # prepare sos
    ys = torch.zeros([1, 128]).type_as(encoder_outputs).long()
    ys[0, 0] = 1

    # yseq: 1xT
    hyp = {'score': 0.0, 'yseq': ys}
    hyps = [hyp]
    ended_hyps = []

    for i in range(maxlen):
        hyps_best_kept = []
        for hyp in hyps:
            ys = hyp['yseq']  # 1 x i
            # -- Prepare masks

            non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1)
            slf_attn_mask = get_subsequent_mask(ys)

            decoder_inputs = {decoder.get_inputs()[0].name: to_numpy(
                ys), decoder.get_inputs()[1].name: to_numpy(encoder_outputs), decoder.get_inputs()[2].name: to_numpy(
                non_pad_mask), decoder.get_inputs()[3].name: to_numpy(slf_attn_mask)}
            dec_output = decoder.run(None, decoder_inputs)[0]

            tgt_word_prj_inputs = {
                tgt_word_prj.get_inputs()[0].name: dec_output[:, i]}
            seq_logit = torch.tensor(
                tgt_word_prj.run(None, tgt_word_prj_inputs)[0])

            local_scores = F.log_softmax(seq_logit, dim=1)
            # topk scores
            local_best_scores, local_best_ids = torch.topk(
                local_scores, beam, dim=1)

            for j in range(beam):
                new_hyp = {}
                new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                new_hyp['yseq'] = copy.deepcopy(hyp['yseq'])
                new_hyp['yseq'][:, i + 1] = int(local_best_ids[0, j])
                # will be (2 x beam) hyps at most
                hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(hyps_best_kept,
                                    key=lambda x: x['score'],
                                    reverse=True)[:beam]
        # end for hyp in hyps
        hyps = hyps_best_kept

        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            for hyp in hyps:
                hyp['yseq'][:, maxlen] = 2

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
        :min(len(ended_hyps), nbest)]
    return nbest_hyps


def recoginze(args):
    encoder = get_onnx_model(args.encoder_path)
    decoder = get_onnx_model(args.decoder_path)
    tgt_word_prj = get_onnx_model(args.tgt_word_prj_path)
    char_list, _, _ = process_dict(args.dict_path)

    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}
    t = time.time()
    for idx, name in enumerate(js.keys(), 1):
        padded_input = kaldi_io.read_mat(js[name]['input'][0]['feat'])  # TxD
        padded_input = build_LFR_features(padded_input, 4, 3)
        padded_input = torch.from_numpy(padded_input).float()

        input_lengths = torch.tensor([padded_input.size(0)], dtype=torch.int)
        padded_input = padded_input.unsqueeze(0)
        padded_input = pad_list(padded_input, 0, max_len=512)
        non_pad_mask = get_non_pad_mask(
            padded_input, input_lengths=input_lengths)
        length = padded_input.size(1)
        slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, length)

        encoder = get_onnx_model(args.encoder_path)
        encoder_inputs = {encoder.get_inputs()[0].name: to_numpy(padded_input), encoder.get_inputs(
        )[1].name: to_numpy(non_pad_mask), encoder.get_inputs()[2].name: to_numpy(slf_attn_mask)}
        encoder_outputs = encoder.run(None, encoder_inputs)
        encoder_outputs = torch.tensor(encoder_outputs, dtype=torch.float)

        nbest_hyps = recognize_beam(
            encoder_outputs[0][0], decoder, tgt_word_prj)

        index = torch.where(nbest_hyps[0]['yseq'][0] == 2)[0]
        nbest_hyps[0]['yseq'] = nbest_hyps[0]['yseq'][:,
                                                      :index + 1][0].cpu().numpy().tolist()
        new_js[name] = add_results_to_json(js[name], nbest_hyps, char_list)
        print('FPS : {:.2f}'.format(idx / (time.time() - t)))

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4,
                           sort_keys=True).encode('utf_8'))


def main():
    args = parser.parse_args()
    recoginze(args)


if __name__ == '__main__':
    main()
