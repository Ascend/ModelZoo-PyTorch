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
        self.random = self.random.astype(np.float16)

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


    def swap_inputs_outputs(self, decoder_inputs, decoder_outputs):

        new_decoder_inputs = [decoder_outputs[0], # decoder_output
                            decoder_outputs[2], # attention_hidden
                            decoder_outputs[3], # attention_cell
                            decoder_outputs[4], # decoder_hidden
                            decoder_outputs[5], # decoder_cell
                            decoder_outputs[6], # attention_weights
                            decoder_outputs[7], # attention_weights_cum
                            decoder_outputs[8], # attention_context
                            decoder_inputs[8],  # memory
                            decoder_inputs[9],  # processed_memory
                            decoder_inputs[10], # mask
                            decoder_inputs[11],
                            decoder_inputs[12]]

        new_decoder_outputs = [decoder_inputs[1], # attention_hidden
                            decoder_inputs[2], # attention_cell
                            decoder_inputs[3], # decoder_hidden
                            decoder_inputs[4], # decoder_cell
                            decoder_inputs[5], # attention_weights
                            decoder_inputs[6], # attention_weights_cum
                            decoder_inputs[7], # attention_context
                            decoder_inputs[0], # decoder_input
                            decoder_outputs[8]]# gate_prediction

        return new_decoder_inputs, new_decoder_outputs


    def sigmoid(self, inx):
        return expit(inx)

    
    def infer(self, batch_size, sequences, sequence_lengths):

        print("Running Tacotron2 Encoder")
        encoder_output = self.encoder_context([sequences, sequence_lengths])

        mask = get_mask_from_lengths(sequence_lengths)
        decoder_inputs = []
        decoder_inputs.append(np.zeros((batch_size, 80), dtype=np.float16))
        decoder_inputs.append(np.zeros((batch_size, 1024), dtype=np.float16))
        decoder_inputs.append(np.zeros((batch_size, 1024), dtype=np.float16))
        decoder_inputs.append(np.zeros((batch_size, 1024), dtype=np.float16))
        decoder_inputs.append(np.zeros((batch_size, 1024), dtype=np.float16))
        decoder_inputs.append(np.zeros((batch_size, sequence_lengths[0]), dtype=np.float16))
        decoder_inputs.append(np.zeros((batch_size, sequence_lengths[0]), dtype=np.float16))
        decoder_inputs.append(np.zeros((batch_size, 512), dtype=np.float16))
        decoder_inputs.append(encoder_output[0])
        decoder_inputs.append(encoder_output[1])
        decoder_inputs.append(mask.numpy())
        decoder_inputs.append(self.random[0])
        decoder_inputs.append(self.random[self.input_random[0]])

        gate_threshold = 0.5
        max_decoder_steps = 2000
        not_finished = torch.ones([batch_size], dtype=torch.int32)
        mel_lengths = torch.zeros([batch_size], dtype=torch.int32)

        print("Running Tacotron2 Decoder")
        exec_seq = 0
        first_iter = True
        while True:
            exec_seq += 1
            decoder_iter_output = self.decoder_context(decoder_inputs)
            decoder_outputs = decoder_iter_output

            if first_iter:
                mel_outputs = np.expand_dims(decoder_outputs[0], 2)
                gate_outputs = np.expand_dims(decoder_outputs[1], 2)
                alignments = np.expand_dims(decoder_outputs[6], 2)
                first_iter = False
            else:
                mel_outputs = np.concatenate((mel_outputs, np.expand_dims(decoder_outputs[0], 2)), 2)
                gate_outputs = np.concatenate((gate_outputs, np.expand_dims(decoder_outputs[1], 2)), 2)
                alignments = np.concatenate((alignments, np.expand_dims(decoder_outputs[6], 2)), 2)

            dec = torch.le(torch.Tensor(self.sigmoid(decoder_outputs[1])), gate_threshold).to(torch.int32).squeeze(1)
            not_finished = not_finished * dec
            mel_lengths += not_finished

            if exec_seq > (sequence_lengths[0] * 6 + sequence_lengths[0] / 2):
                print("Stopping after", exec_seq,"decoder steps")
                break

            if torch.sum(not_finished) == 0:
                print("Stopping after", mel_outputs.shape[2],"decoder steps")
                break
            if mel_outputs.shape[2] == max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_inputs, decoder_iter_output = self.swap_inputs_outputs(decoder_inputs, decoder_iter_output)
            decoder_inputs[11] = self.random[exec_seq]
            decoder_inputs[12] = self.random[self.input_random[exec_seq]]

        mel_outputs_length = mel_outputs.shape[2]
        mel_outputs_padded = np.zeros((batch_size, 80, max_decoder_steps), dtype=np.float16)
        mel_outputs_padded[:,:,:mel_outputs_length] = mel_outputs

        mel_outputs_postnet = self.postnet_context(mel_outputs_padded)
        mel_outputs_postnet = mel_outputs_postnet[0][:,:,:mel_outputs_length]

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
            mel, mel_lengths = tacotron2.infer(batch_size, sequences, sequence_lengths)

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
    print("tacotron2_items_per_sec: {}".format(all_mels/all_time))


if __name__ == "__main__":
    main()
