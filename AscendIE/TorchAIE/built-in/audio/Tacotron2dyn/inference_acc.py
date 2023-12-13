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


import os
import platform
import argparse

import numpy as np
import torch
import torch_aie
import onnxruntime as rt
from scipy.io.wavfile import write
from scipy.special import expit
from tacotron2.text import text_to_sequence

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


def sigmoid(inx):
    return expit(inx)

def update_decoder_inputs(decoder_inputs, decoder_outputs):
    new_decoder_inputs = [
        decoder_outputs[0],  # decoder_output
        decoder_outputs[2],  # attention_hidden
        decoder_outputs[3],  # attention_cell
        decoder_outputs[4],  # decoder_hidden
        decoder_outputs[5],  # decoder_cell
        decoder_outputs[6],  # attention_weights
        decoder_outputs[7],  # attention_weights_cum
        decoder_outputs[8],  # attention_context
        decoder_inputs[8],   # memory
        decoder_inputs[9],   # processed_memory
        decoder_inputs[10],  # mask
    ]

    return new_decoder_inputs


def inference_tacotron(seqs, seq_lens, encoder_model, decoder_model, posnet_model):
    #infer encoder
    seqs = seqs.to("npu:0")
    seq_lens = seq_lens.to("npu:0")
    encoder_output = encoder_model(seqs, seq_lens)

    #inder decoder
    seqs = seqs.cpu()
    seq_lens = seq_lens.cpu()
    seq_lens = seq_lens.numpy()


    mask_from_length_tensor = get_mask_from_lengths(seq_lens).float().numpy()
    mask_from_length_tensor = torch.from_numpy(mask_from_length_tensor)


    #11 decoder inputs
    decoder_inputs = []
    decoder_inputs.append(torch.zeros((args.batch_size, 80), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, 1024), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, 1024), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, 1024), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, 1024), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, seq_lens[0]), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, seq_lens[0]), dtype=torch.float32))
    decoder_inputs.append(torch.zeros((args.batch_size, 512), dtype=torch.float32))
    decoder_inputs.append(encoder_output[0])
    decoder_inputs.append(encoder_output[1])
    decoder_inputs.append(mask_from_length_tensor)


    gate_threshold = 0.5
    max_decoder_steps = 2000
    first_iter = True
    not_finished = torch.ones([args.batch_size], dtype=torch.int32)
    mel_lengths = torch.zeros([args.batch_size], dtype=torch.int32)

    print("Starting run Tacotron2 decoder ……")
    exec_seq = 0
    while True:
        exec_seq += 1

        
        input_1 = decoder_inputs[0].to("npu:0")
        input_2 = decoder_inputs[1].to("npu:0")
        input_3 = decoder_inputs[2].to("npu:0")
        input_4 = decoder_inputs[3].to("npu:0")
        input_5 = decoder_inputs[4].to("npu:0")
        input_6 = decoder_inputs[5].to("npu:0")
        input_7 = decoder_inputs[6].to("npu:0")
        input_8 = decoder_inputs[7].to("npu:0")
        input_9 = decoder_inputs[8].to("npu:0")
        input_10 = decoder_inputs[9].to("npu:0")
        input_11 = decoder_inputs[10].to("npu:0")
        

        decoder_iter_output_npu = decoder_model(input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11 )
        decoder_iter_output = []
        decoder_iter_output.append(decoder_iter_output_npu[0].cpu())
        decoder_iter_output.append(decoder_iter_output_npu[1].cpu())
        decoder_iter_output.append(decoder_iter_output_npu[2].cpu())
        decoder_iter_output.append(decoder_iter_output_npu[3].cpu())
        decoder_iter_output.append(decoder_iter_output_npu[4].cpu())
        decoder_iter_output.append(decoder_iter_output_npu[5].cpu())
        decoder_iter_output.append(decoder_iter_output_npu[6].cpu())
        decoder_iter_output.append(decoder_iter_output_npu[7].cpu())
        decoder_iter_output.append(decoder_iter_output_npu[8].cpu())

        decoder_inputs = update_decoder_inputs(decoder_inputs, decoder_iter_output)

        if first_iter:
            mel_outputs = np.expand_dims(decoder_iter_output[0], 2)
            first_iter = False
        else:
            mel_outputs = np.concatenate((mel_outputs, np.expand_dims(decoder_iter_output[0], 2)), 2)

        # decide whether stop decoder or not
        dec = torch.le(torch.Tensor(sigmoid(decoder_iter_output[1])), gate_threshold).to(torch.int32).squeeze(1)
        not_finished = not_finished * dec
        mel_lengths += not_finished

        if exec_seq > (seq_lens[0] * 6 + seq_lens[0] / 2):
            print("Warning! exec_seq > seq_lens, Stop after ", exec_seq, " decoder steps")
            break
        if mel_outputs.shape[2] == max_decoder_steps:
            print("Warning! Reach max decoder steps", max_decoder_steps)
            break
        if torch.sum(not_finished) == 0:
            print("Finished! Stop after ", mel_outputs.shape[2], " decoder steps")
            break
            
    #infer posnet
    mel_outputs = torch.from_numpy(mel_outputs).to("npu:0")

    #padmel_outputs to ensure posnet get static shape input 
    _, _, x = mel_outputs.shape
    target_shape = [1, 80, 620]
    if (x < target_shape[2]):
        pad_size = target_shape[2] - x
        padding = (0, pad_size)
        mel_outputs = mel_outputs.cpu()
        mel_outputs = torch.nn.functional.pad(mel_outputs, padding, "constant", 0)
        mel_outputs = mel_outputs.to("npu:0")

    mel_outputs_postnet = posnet_model(mel_outputs)
    mel_outputs_postnet = mel_outputs_postnet.cpu()

    print("Tacotron2 infer success")
    return mel_outputs_postnet, mel_lengths

def inference_waveglow(waveglow_model, tacotron2_output, mel_lengths, batch_size):

    mel = torch.randn(batch_size, 80, 620)

    stride = 256 # value from waveglow upsample
    n_group = 8
    z_size2 = (mel.size(2)*stride)//n_group
    z = torch.randn(batch_size, n_group, z_size2)

    tacotron2_output = tacotron2_output.to("npu:0")
    z = z.to("npu:0")

    waveglow_output = waveglow_model(tacotron2_output, z)
    waveglow_output = waveglow_output.cpu()

    return waveglow_output

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_model_path', type=str, required=True)
    parser.add_argument('--decoder_model_path', type=str, required=True)
    parser.add_argument('--posnet_model_path', type=str, required=True)
    parser.add_argument('--waveglow_model_path', type=str, required=True)


    parser.add_argument('-i', '--input', type=str, required=True,
                        help='input text')
    parser.add_argument('-o', '--output', required=False, default="output/audio", type=str,
                       help='output folder to save autio')
    parser.add_argument('-bs', '--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--max_input_len', default=256, type=int, help='max input len') 
    parser.add_argument('--gen_wave', action='store_true', help='generate wav file')
    parser.add_argument('--stft_hop_length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')

    args = parser.parse_args()

    encoder_model = torch.jit.load(args.encoder_model_path)
    decoder_model = torch.jit.load(args.decoder_model_path)
    posnet_model = torch.jit.load(args.posnet_model_path)
    waveglow_model = torch.jit.load(args.waveglow_model_path)


    torch_aie.set_device(0)
    os.makedirs(args.output, exist_ok=True)

    # load wav_texts data
    wav_names, wav_texts = load_wav_texts(args.input)

    while args.batch_size <= len(wav_texts):
        # data preprocess (prepare batch & load)
        batch_names, batch_texts = prepare_batch_wav(args.batch_size, wav_names, wav_texts, args.max_input_len)
        seqs, seq_lens, batch_names_new = prepare_input_sequence(batch_names, batch_texts)
        if seqs == '':
            print("Invalid input!")
            break
        seqs = seqs.to(torch.int64).numpy()
        seq_lens = seq_lens.to(torch.int32).numpy()

        seqs = torch.from_numpy(seqs)
        seq_lens = torch.from_numpy(seq_lens)
        #infer tacotron
        tacotron2_output, mel_lengths = inference_tacotron(seqs, seq_lens, encoder_model.eval(), decoder_model.eval(), posnet_model.eval())
        # generate wave file
        if args.gen_wave:
            waveglow_output = inference_waveglow(waveglow_model.eval(), tacotron2_output, mel_lengths, args.batch_size).numpy().astype(np.float32)
            for i, audio in enumerate(waveglow_output):
                audio = audio[:mel_lengths[i] * args.stft_hop_length]
                audio = audio / np.amax(np.absolute(audio))
                audio_path = os.path.join(args.output, batch_names_new[i] + ".wav")
                write(audio_path, args.sampling_rate, audio)

