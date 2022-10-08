# Copyright 2018 NVIDIA Corporation. All Rights Reserved.
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
# ============================================================================
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
import sys
import wave
import difflib
import numpy as np
import time
import torch
import argparse
import logging

from scipy.io.wavfile import write
from scipy.special import expit
from torch import jit
from inference import MeasureTime
from onnx_infer import Waveglow
from data_process import *
from acl_net import Net
import acl


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='input text')
    parser.add_argument('-o', '--output', required=False, default="output/",
                        help='output folder')
    parser.add_argument('--log-file', type=str, default='pyacl_log.json',
                        help='Filename for logging')
    parser.add_argument('-bs', '--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('-max_input_len', default=128, type=int,
                        help='max input len')
    parser.add_argument('-max_decode_iter', default=20, type=int,
                        help='max decode times')
    parser.add_argument('--device_id', default=0, type=int,
                        help='device id')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')                        
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')

    return parser


class Tacotron2():
    def __init__(self, device_id):
        self.max_decoder_steps = 2000
        self.random = np.random.rand(self.max_decoder_steps+1, 256)
        self.random = self.random.astype(np.float32)

        self.input_random = np.random.randint(1, self.max_decoder_steps, size=(self.max_decoder_steps))

        ret = acl.init()
        assert ret == 0
        ret = acl.rt.set_device(device_id)
        assert ret == 0
        context, ret = acl.rt.create_context(device_id)
        assert ret == 0
        self.device_id = device_id

        self.encoder_context = Net(context, device_id=self.device_id, 
                                model_path="output/encoder_static.om", first=True)
        self.decoder_context = Net(context, device_id=self.device_id, 
                                model_path="output/decoder_static.om", first=False)                                
        self.postnet_context = Net(context, device_id=self.device_id, 
                                model_path="output/postnet_static.om", first=False)

    def __del__(self):
        del self.encoder_context
        del self.decoder_context
        del self.postnet_context

        ret = acl.rt.reset_device(self.device_id)
        assert ret == 0
        context, ret = acl.rt.get_context()
        assert ret == 0
        ret = acl.rt.destroy_context(context)
        assert ret == 0
        ret = acl.finalize()
        assert ret == 0

    def infer(self, batch_size, sequences, sequence_lengths, max_decode_iter):

        print("Running Tacotron2 Encoder")
        mask = get_mask_from_lengths(sequence_lengths) 
        mask = mask.numpy()
        decoder_input = np.zeros((batch_size, 80), dtype=np.float32)
        attention_hidden = np.zeros((batch_size, 1024), dtype=np.float32)
        attention_cell = np.zeros((batch_size, 1024), dtype=np.float32)
        decoder_hidden = np.zeros((batch_size, 1024), dtype=np.float32)
        decoder_cell = np.zeros((batch_size, 1024), dtype=np.float32)
        attention_weights = np.zeros((batch_size, sequence_lengths[0]), dtype=np.float32)
        attention_weights_cum = np.zeros((batch_size, sequence_lengths[0]), dtype=np.float32)
        attention_context = np.zeros((batch_size, 512), dtype=np.float32)

        not_finished = np.ones((batch_size,), dtype=np.int32)
        mel_lengths = np.zeros((batch_size,), dtype=np.int32)
        mel_output_input = np.zeros((batch_size, 80, 1), dtype=np.float32)
        gate_output_input = np.zeros((batch_size, 1, 1), dtype=np.float32)
        encoder_output = self.encoder_context([sequences, sequence_lengths, decoder_input, attention_hidden,
                                               attention_cell, decoder_hidden, decoder_cell, attention_weights,
                                               attention_weights_cum, attention_context, mask, not_finished,
                                               mel_lengths])

        gate_threshold = 0.5
        max_decoder_steps = 2000

        print("Running Tacotron2 Decoder")
        mel_output_0 = encoder_output[-4]
        gate_output_0 = encoder_output[-3]
        mel_outputs = mel_output_0
        gate_outputs = gate_output_0
        decoder_output = encoder_output
        for i in range(max_decode_iter):
            decoder_output = self.decoder_context(decoder_output)
            decoder_output_0_11 = decoder_output[:-4]
            mel_output_cat = decoder_output[-4]
            gate_output_cat = decoder_output[-3]
            decoder_output_14_15 = decoder_output[-2:]
            mel_outputs = np.concatenate((mel_outputs, mel_output_cat), 2)
            gate_outputs = np.concatenate((gate_outputs, gate_output_cat), 2)
            decoder_output = decoder_output_0_11 + [mel_output_input, gate_output_input] + decoder_output_14_15
            not_finished = decoder_output[-2]
            if np.sum(not_finished) == 0:
                break

        mel_outputs_length = mel_outputs.shape[2]
        mel_outputs_padded = np.zeros((batch_size, 80, max_decoder_steps), dtype=np.float32)
        mel_outputs_padded[:, :, :mel_outputs_length] = mel_outputs

        mel_outputs_postnet = self.postnet_context(mel_outputs_padded)
        mel_outputs_postnet = mel_outputs_postnet[0][:, :, :mel_outputs_length]
        mel_lengths = decoder_output[-1]
        mel_lengths = torch.from_numpy(mel_lengths)
        print("Tacotron2 Postnet done")
        return mel_outputs_postnet, mel_lengths


def main():
    parser = argparse.ArgumentParser(
        description='ONNX Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    texts = []
    batch_size = args.batch_size

    try:
        name_list, value_list = read_file(args.input)
    except Exception as e:
        print("Could not read file")
        sys.exit(1)

    batch_num = 0
    from collections import defaultdict
    cost_time = defaultdict(float)
    offset = 0
    tacotron2 = Tacotron2(device_id=args.device_id)
    data_procss = DataProcess(args.max_input_len, False, 0)
    waveglow = Waveglow("output/waveglow.onnx")
    all_time = 0
    all_mels = 0

    while batch_size <= len(value_list):
        measurements = {}
        if batch_size == 1 and len(value_list[0]) < args.max_input_len:
            print("input text less max input size")
            break
        
        batch_texts, batch_names = data_procss.prepare_batch_meta(batch_size, value_list, name_list)
        offset += batch_size
        batch_num += 1

        sequences, sequence_lengths, batch_names_new = data_procss.prepare_input_sequence(batch_texts, 
                                                                batch_names)
        if sequences == '' or len(batch_texts[0]) < args.max_input_len:
            print("input text less max input size")
            break

        sequences = sequences.to(torch.int64).numpy()
        sequence_lengths = sequence_lengths.to(torch.int32).numpy()

        with MeasureTime(measurements, "tacotron2_latency", cpu_run=True):
            mel, mel_lengths = tacotron2.infer(batch_size, sequences, sequence_lengths, args.max_decode_iter)

        if args.device_id == 0:
            waveglow_output = waveglow.infer(mel)
            waveglow_output = waveglow_output.astype(np.float32)

            for i, audio in enumerate(waveglow_output):
                audio = audio[:mel_lengths[i] * args.stft_hop_length]
                audio = audio / np.amax(np.absolute(audio))
                audio_path = args.output + batch_names_new[i] + ".wav"
                write(audio_path, args.sampling_rate, audio)

        num_mels = mel.shape[0] * mel.shape[2]
        all_mels += num_mels
        all_time += measurements["tacotron2_latency"]
    perf = all_mels/all_time
    resstr = "perf: {}\n".format(perf)
    with open("results.txt", "a") as resfile:
        resfile.write(resstr)


if __name__ == "__main__":
    main()
