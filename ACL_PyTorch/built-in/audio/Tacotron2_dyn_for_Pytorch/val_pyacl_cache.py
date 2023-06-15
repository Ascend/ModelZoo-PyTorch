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

import argparse
import difflib
import logging
import os
import platform
import sys

import wave
import numpy as np
import torch
import onnxruntime as rt
import acl
from torch import jit
from scipy.io.wavfile import write
from scipy.special import expit
from pyacl.acl_infer import AclNet, init_acl, release_acl

from tacotron2.text import text_to_sequence
from inference import MeasureTime


def parse_args():
    system = platform.system().lower()
    machine = platform.machine()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='input text')
    parser.add_argument('-o', '--output', required=False, default="output/audio", type=str,
                        help='output folder to save autio')
    parser.add_argument('--encoder', default=f'./output/om/encoder_dyn_{system}_{machine}.om', type=str,
                        help='load encoder model')
    parser.add_argument('--decoder', default=f'./output/om/decoder_iter_dyn_{system}_{machine}.om', type=str,
                        help='load decoder model')
    parser.add_argument('--postnet', default=f'./output/om/postnet_dyn_{system}_{machine}.om', type=str,
                        help='load postnet model')
    parser.add_argument('-bs', '--batch_size', default=1, type=int,
                        help='Batch size')
    parser.add_argument('--max_input_len', default=256, type=int,
                        help='max input len')
    parser.add_argument('--gen_wave', action='store_true',
                        help='generate wav file')
    parser.add_argument('--waveglow', default='./output/onnx/waveglow.onnx', type=str,
                        help='load waveglow model')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--device_id', default=0, type=int,
                        help='device id')
    parser.add_argument('--use_dynamic_data_buffer', action='store_true',
                        help='debug mode')
    return parser.parse_args()


def pad_sequences(batch_seqs, batch_names):
    import copy
    batch_copy = copy.deepcopy(batch_seqs)
    for i in range(len(batch_copy)):
        if len(batch_copy[i]) > args.max_input_len:
            batch_seqs[i] = batch_seqs[i][:args.max_input_len]

    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch_seqs]), dim=0, descending=True)

    text_padded = torch.LongTensor(len(batch_seqs), input_lengths[0])
    text_padded.zero_()
    text_padded[0][:] += torch.IntTensor(text_to_sequence('.', ['english_cleaners'])[:])
    names_new = []
    for i in range(len(ids_sorted_decreasing)):
        text = batch_seqs[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text
        names_new.append(batch_names[ids_sorted_decreasing[i]])

    return text_padded, input_lengths, names_new


def prepare_input_sequence(batch_names, batch_texts):
    batch_seqs = []
    for i, text in enumerate(batch_texts):
        batch_seqs.append(torch.IntTensor(text_to_sequence(text, ['english_cleaners'])[:]))

    text_padded, input_lengths, names_new = pad_sequences(batch_seqs, batch_names)

    text_padded = text_padded.long()
    input_lengths = input_lengths.long()

    return text_padded, input_lengths, names_new


def prepare_batch_wav(batch_size, wav_names, wav_texts, max_input):
    batch_texts = []
    batch_names = []
    for i in range(batch_size):
        if i == 0:
            batch_names.append(wav_names.pop(0))
            batch_texts.append(wav_texts.pop(0))
        else:
            batch_names.append(wav_names.pop())
            batch_texts.append(wav_texts.pop())
    if len(batch_texts[0]) < max_input:
        batch_texts[0] += ' a'
    return batch_names, batch_texts


def load_wav_texts(input_file):
    metadata_dict = {}
    if input_file.endswith('.csv'):
        with open(input_file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                metadata_dict[line.strip().split('|')[0]] = line.strip().split('|')[-1]
    elif input_file.endswith('.txt'):
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                wav_name = line.split('|')[0].split('/', 2)[2].split('.')[0]
                wav_text = line.split('|')[1]
                metadata_dict[wav_name] = wav_text
    else:
        print("file is not support")

    wavs = sorted(metadata_dict.items(), key=lambda value: len(value[1]), reverse=True)
    wav_names = [wav[0] for wav in wavs]
    wav_texts = [wav[1].strip() for wav in wavs]

    return wav_names, wav_texts


def get_mask_from_lengths(lengths):
    lengths_tensor = torch.LongTensor(lengths)
    max_len = torch.max(lengths_tensor).item()
    ids = torch.arange(0, max_len, device=lengths_tensor.device, dtype=lengths_tensor.dtype)
    mask = (ids < lengths_tensor.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask


class Tacotron2():
    def __init__(self, encoder, decoder, postnet, device_id, cost_time=0,
                 encoder_cost_time=0, decoder_cost_time=0, postnet_cost_time=0,
                 use_dynamic_data_buffer=False):
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet
        self.use_dynamic_data_buffer = use_dynamic_data_buffer

        self.cost_time = cost_time
        self.encoder_cost_time = encoder_cost_time
        self.decoder_cost_time = decoder_cost_time
        self.postnet_cost_time = postnet_cost_time
        self.max_decoder_steps = 2000

        np.random.seed(1)
        self.random = np.random.rand(self.max_decoder_steps + 1, 256)
        self.random = self.random.astype(np.float16)

    def update_decoder_inputs(self, decoder_inputs, decoder_outputs):
        new_decoder_inputs = [
            decoder_outputs[0],  # decoder_output
            decoder_outputs[2],  # attention_hidden
            decoder_outputs[3],  # attention_cell
            decoder_outputs[4],  # decoder_hidden
            decoder_outputs[5],  # decoder_cell
            decoder_outputs[6],  # attention_weights
            decoder_outputs[7],  # attention_weights_cum
            decoder_outputs[8],  # attention_context
            decoder_inputs[8],  # memory
            decoder_inputs[9],  # processed_memory
            decoder_inputs[10],  # mask
        ]

        return new_decoder_inputs

    def sigmoid(self, inx):
        return expit(inx)

    def infer_tacotron2(self, seqs, seq_lens, measurements):
        print("Starting run Tacotron2 encoder ……")
        with MeasureTime(measurements, "tacotron2_encoder_time", cpu_run=True):
            encoder_output, encoder_t = self.encoder([seqs, seq_lens])
        self.encoder_cost_time = measurements["tacotron2_encoder_time"]

        ##### init decoder_inputs
        decoder_inputs = []
        decoder_inputs.append(np.zeros((args.batch_size, 80), dtype=np.float32))
        decoder_inputs.append(np.zeros((args.batch_size, 1024), dtype=np.float32))
        decoder_inputs.append(np.zeros((args.batch_size, 1024), dtype=np.float32))
        decoder_inputs.append(np.zeros((args.batch_size, 1024), dtype=np.float32))
        decoder_inputs.append(np.zeros((args.batch_size, 1024), dtype=np.float32))
        decoder_inputs.append(np.zeros((args.batch_size, seq_lens[0]), dtype=np.float32))
        decoder_inputs.append(np.zeros((args.batch_size, seq_lens[0]), dtype=np.float32))
        decoder_inputs.append(np.zeros((args.batch_size, 512), dtype=np.float32))
        decoder_inputs.append(encoder_output[0])
        decoder_inputs.append(encoder_output[1])
        decoder_inputs.append(get_mask_from_lengths(seq_lens).numpy())

        gate_threshold = 0.5
        max_decoder_steps = 2000
        first_iter = True
        not_finished = torch.ones([args.batch_size], dtype=torch.int32)
        mel_lengths = torch.zeros([args.batch_size], dtype=torch.int32)
        not_finished_flag = False
        print("Starting run Tacotron2 decoder ……")
        decoder_iter_output, exe_t = self.decoder(decoder_inputs, first_step=True, end_step=False)
        
        decoder_inputs = self.update_decoder_inputs(decoder_inputs,decoder_iter_output)
        mel_outputs = np.expand_dims(decoder_iter_output[0], 2)
        # decide whether stop decoder or not
        dec = torch.le(torch.Tensor(self.sigmoid(decoder_iter_output[1])), gate_threshold).to(torch.int32).squeeze(1)
        not_finished = not_finished * dec
        mel_lengths += not_finished
        
        for i in range(1, max_decoder_steps-1):
            with MeasureTime(measurements, "tacotron2_decoder_time", cpu_run=True):
                decoder_iter_output, exe_t = self.decoder(decoder_inputs, first_step=False, end_step=False)
            self.decoder_cost_time += measurements["tacotron2_decoder_time"]

            decoder_inputs = self.update_decoder_inputs(decoder_inputs, decoder_iter_output)
            mel_outputs = np.concatenate((mel_outputs, np.expand_dims(decoder_iter_output[0], 2)), 2)
            
            # decide whether stop decoder or not
            dec = torch.le(torch.Tensor(self.sigmoid(decoder_iter_output[1])), gate_threshold).to(torch.int32).squeeze(1)
            not_finished = not_finished * dec
            mel_lengths += not_finished
            
            
            if torch.sum(not_finished) == 0:
                not_finished_flag = True
                print("Finished! Stop after ", mel_outputs.shape[2], " decoder steps")
                break
        
        if not_finished_flag:
            decoder_output, exe_t = self.decoder.get_final_result()
        else:
            decoder_output, exe_t = self.decoder(decoder_inputs, first_step=False, end_step=True)
            mel_outputs = np.concatenate((mel_outputs, np.expand_dims(decoder_iter_output[0], 2)), 2)

        print("Starting run Tacotron2 postnet ……")
        with MeasureTime(measurements, "tacotron2_postnet_time", cpu_run=True):
            mel_outputs_postnet, postnet_t = self.postnet(mel_outputs)
        self.postnet_cost_time = measurements["tacotron2_postnet_time"]
        mel_outputs_postnet = mel_outputs_postnet[0]

        print("Tacotron2 infer success")
        self.cost_time = (self.encoder_cost_time + self.decoder_cost_time + self.postnet_cost_time)

        return mel_outputs_postnet, mel_lengths

    def infer_waveglow(self, waveglow, mel):
        mel = np.array(mel).astype(np.float32)
        mel_size = mel.shape[2]
        batch_size = mel.shape[0]
        stride = 256
        n_group = 8
        z_size = mel_size * stride // n_group
        z = np.random.randn(batch_size, n_group, z_size).astype(np.float32)

        sess, input_name, output_name = self.onnxruntime_init(waveglow)
        waveglow_output, _ = self.onnxruntime_run(sess, input_name, output_name, [mel, z])
        waveglow_output = np.reshape(waveglow_output, (args.batch_size, -1))  # batch_size, seq_len
        return waveglow_output

    def onnxruntime_init(self, model):
        sess = rt.InferenceSession(model)
        input_name = []
        for n in sess.get_inputs():
            input_name.append(n.name)
        output_name = []
        for n in sess.get_outputs():
            output_name.append(n.name)
        return sess, input_name, output_name

    def onnxruntime_run(self, sess, input_name, output_name, input_data):
        res_buff = []
        succ = True

        res = sess.run(None, {input_name[i]: input_data[i] for i in range(len(input_name))})
        for i, x in enumerate(res):
            out = np.array(x)
            res_buff.append(out)

        return res_buff, succ


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # load wav_texts data
    wav_names, wav_texts = load_wav_texts(args.input)

    # load model
    init_acl(args.device_id)
    encoder = AclNet(
        model_path=args.encoder,
        device_id=args.device_id,
        input_data_shape=100000 * args.batch_size,
        output_data_shape=[624288 * 4, 624288 * 4, 4] * args.batch_size
    )
    print("load encoder success")
    decoder = AclNet(
        model_path=args.decoder,
        device_id=args.device_id,
        input_data_shape=100000 * args.batch_size,
        out_to_in={0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7},
        out_idx=[0, 1],
        pin_input=[8, 9, 10],
        output_data_shape=20000 * args.batch_size
    )
    print("load decoder success")
    postnet = AclNet(
        model_path=args.postnet,
        device_id=args.device_id,
        input_data_shape=100000 * args.batch_size,
        output_data_shape=640000 * args.batch_size
    )
    print("load postnet success")
    tacotron2 = Tacotron2(encoder, decoder, postnet,
                          args.device_id, use_dynamic_data_buffer=args.use_dynamic_data_buffer)

    all_time = 0
    all_mels = 0
    while args.batch_size <= len(wav_texts):
        # data preprocess (prepare batch & load)
        batch_names, batch_texts = prepare_batch_wav(args.batch_size, wav_names, wav_texts, args.max_input_len)
        seqs, seq_lens, batch_names_new = prepare_input_sequence(batch_names, batch_texts)
        if seqs == '':
            print("Invalid input!")
            break
        seqs = seqs.to(torch.int64).numpy()
        seq_lens = seq_lens.to(torch.int32).numpy()

        # inference Tacotron2
        measurements = {}
        with MeasureTime(measurements, "tacotron2_latency", cpu_run=True):
            tacotron2_output, mel_lengths = tacotron2.infer_tacotron2(seqs, seq_lens, measurements)

        # generate wave file
        if args.gen_wave:
            with MeasureTime(measurements, "waveglow_time", cpu_run=True):
                waveglow_output = tacotron2.infer_waveglow(args.waveglow, tacotron2_output).astype(np.float32)

            for i, audio in enumerate(waveglow_output):
                audio = audio[:mel_lengths[i] * args.stft_hop_length]
                audio = audio / np.amax(np.absolute(audio))
                audio_path = os.path.join(args.output, batch_names_new[i] + ".wav")
                write(audio_path, args.sampling_rate, audio)

        # compute cost time
        num_mels = tacotron2_output.shape[0] * tacotron2_output.shape[2]
        all_mels += num_mels
        all_time += measurements["tacotron2_latency"]
    
    encoder.release_model()
    decoder.release_model()
    postnet.release_model()
    release_acl(args.device_id)
    print(f"tacotron2_items_per_sec: {all_mels / all_time}")
