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
import json
import argparse

import torch
import torch.nn as nn
import yaml
import numpy as np
from gener_core.mod_modify.interface import AttrType as AT
from gener_core.mod_modify.onnx_graph import OXGraph

import hifigan
from model import FastSpeech2
from model.modules import VarianceAdaptor
from transformer import Decoder
from utils.tools import get_mask_from_lengths


class Encoder(nn.Module):
    def __init__(self, fastspeech2):
        super(Encoder, self).__init__()
        self.encoder = fastspeech2.encoder

    def forward(self, src_seq, src_mask):
        output = self.encoder(src_seq, src_mask)

        return output


class _Bucketize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, boundaries):
        y = torch.randint(10, 240, x.size())
        return y

    @staticmethod
    def symbolic(g, x, boundaries):
        y = g.op("Bucketize", x, boundaries_f=boundaries)
        return y


def _bucketize(x, boundaries):
    return _Bucketize.apply(x, boundaries)


def modify_if(input_onnx, output_onnx):
    '''
      add  —— gather —— equal —— if —— where
                         condition  ——
                                 x  ——
      add‘ ————————— squeeze ————————— where’
    '''

    mod = OXGraph(input_onnx)
    io_map = mod.get_net_in_out_map()

    # get operator to modify
    if_node = mod.get_nodes_by_optype("If")[0]
    if if_node:
        equal_node = mod.get_node(if_node.input_name[0])
        gather_node = mod.get_node(equal_node.input_name[0])
        shape_node = mod.get_node(gather_node.input_name[0])

        # add squeeze_node, update input
        squeeze_node = mod.add_new_node("Squeeze", "Squeeze",
                                        {"axes": (AT.LIST_INT, [2])})
        add_node = mod.get_node(shape_node.input_name[0])
        squeeze_node.set_input_node(0, [add_node])

        # update input of where_node(origin cast/x, new squeeze)
        where_node = mod.get_node(io_map.get(if_node.name)[0])
        cast_node = mod.get_node(where_node.input_name[0])
        x_node = mod.get_node(where_node.input_name[1])
        where_node.set_input_node(0, [cast_node, x_node, squeeze_node])

        # delete useless node, save modified model
        mod.node_remove([shape_node.name, gather_node.name, equal_node.name, if_node.name])
        mod.save_new_model(output_onnx)
    else:
        pass


class VarianceAdaptorSim(VarianceAdaptor):
    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptorSim, self).__init__(preprocess_config, model_config)

    def get_pitch_embedding(self, x, mask, control):
        prediction = self.pitch_predictor(x, mask)
        prediction = prediction * control
        embedding = self.pitch_embedding(_bucketize(prediction, self.pitch_bins.numpy()))
        return prediction, embedding

    def get_energy_embedding(self, x, mask, control):
        prediction = self.energy_predictor(x, mask)
        prediction = prediction * control
        embedding = self.energy_embedding(_bucketize(prediction, self.energy_bins.numpy()))
        return prediction, embedding

    def forward(self, x, src_mask, p_control=1.0, e_control=1.0, d_control=1.0):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        duration_rounded = torch.clamp((torch.round(torch.exp(log_duration_prediction) - 1) * d_control), min=0)

        pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, src_mask, p_control)
        x = x + pitch_embedding

        energy_prediction, energy_embedding = self.get_energy_embedding(x, src_mask, e_control)
        x = x + energy_embedding

        return (x, duration_rounded)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class DecoderExt(Decoder):
    def __init__(self, config):
        super(DecoderExt, self).__init__(config)
        n_position = config["max_seq_len"] * 2
        d_word_vec = config["transformer"]["decoder_hidden"]

        self.position_enc_ext = get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0)

    def forward(self, enc_seq, mask):
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # Forward
        max_len = min(max_len, self.max_seq_len * 2)

        # Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        dec_output = enc_seq[:, :max_len, :] + self.position_enc_ext[:, :max_len, :].expand(batch_size, -1, -1)
        mask = mask[:, :max_len]
        slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(dec_output, mask=mask, slf_attn_mask=slf_attn_mask)

        return dec_output


class Postnet(nn.Module):
    def __init__(self, fastspeech2):
        super(Postnet, self).__init__()
        self.mel_linear = fastspeech2.mel_linear
        self.postnet = fastspeech2.postnet

    def forward(self, dec_output):
        output_tmp = self.mel_linear(dec_output)
        output = self.postnet(output_tmp) + output_tmp

        return output


class Vocoder(nn.Module):
    def __init__(self, vocoder, preprocess_config):
        super(Vocoder, self).__init__()
        self.vocoder = vocoder
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]

    def forward(self, mel_output):
        mel_predictions = mel_output.transpose(1, 2)
        wavs = self.vocoder(mel_predictions).squeeze(1)

        return wavs


def get_model(args, configs):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config)
    model.variance_adaptor = VarianceAdaptorSim(preprocess_config, model_config)
    model.decoder = DecoderExt(model_config)
    if args.restore_step:
        ckpt_path = os.path.join(train_config["path"]["ckpt_path"], f"{args.restore_step}.pth.tar")
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt["model"])

    model.eval()
    model.requires_grad_ = False

    return model


def get_vocoder(vocoder_name):
    if vocoder_name != "HiFi-GAN":
        print("support HiFi-GAN only")

    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=torch.device('cpu'))
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()

    return vocoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, help="pretrained model")
    parser.add_argument("--output", type=str, default="output/onnx", help="Path to export FastSpeech2 onnx model")
    parser.add_argument("-p", "--preprocess_config", type=str, help="path to preprocess.yaml")
    parser.add_argument("-m", "--model_config", type=str, help="path to model.yaml")
    parser.add_argument("-t", "--train_config", type=str, help="path to train.yaml")
    parser.add_argument("--pitch_control", type=float, default=1.0,
                        help="control the pitch of the whole utterance, larger value for higher pitch")
    parser.add_argument("--energy_control", type=float, default=1.0,
                        help="control the energy of the whole utterance, larger value for larger volume")
    parser.add_argument("--duration_control", type=float, default=1.0,
                        help="control the speed of the whole utterance, larger value for slower speaking rate")
    args = parser.parse_args()

    # read config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # load model
    fastspeech2 = get_model(args, configs)
    vocoder = get_vocoder("HiFi-GAN")

    # create dummy input data
    texts = torch.randint(low=0, high=148, size=(1, 50), dtype=torch.long)
    text_lens = torch.IntTensor([texts.size(1)])
    src_masks = get_mask_from_lengths(text_lens)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print("Starting export encoder ……")
    encoder = Encoder(fastspeech2)
    encoder.eval()
    encoder_input = (texts, src_masks)
    enc_output = encoder(*encoder_input)

    torch.onnx.export(encoder, encoder_input, os.path.join(args.output, "encoder.onnx"),
                      opset_version=11, do_constant_folding=True,
                      input_names=["texts", "src_masks"],
                      output_names=["enc_output"],
                      dynamic_axes={"texts": {0: "batch_size", 1: "max_src_len"},
                                    "src_masks": {0: "batch_size", 1: "max_src_len"}})

    print("Starting export variance_adaptor ……")
    variance_adaptor = fastspeech2.variance_adaptor
    variance_adaptor.eval()
    adaptor_input = (enc_output, src_masks, args.pitch_control, args.energy_control, args.duration_control)
    (output, duration_rounded) = variance_adaptor(*adaptor_input)

    torch.onnx.export(variance_adaptor, adaptor_input, os.path.join(args.output, "variance_adaptor.onnx"),
                      opset_version=11, do_constant_folding=True,
                      input_names=["enc_output", "src_masks", "p_control", "e_control", "d_control"],
                      output_names=["output", "duration_rounded"],
                      dynamic_axes={"enc_output": {0: "batch_size", 1: "max_src_len"},
                                    "src_masks": {0: "batch_size", 1: "max_src_len"}})

    # if If_node exist in variance_adaptor, modify to optimize performance
    variance_adaptor_onnx = os.path.join(args.output, "variance_adaptor.onnx")
    modify_if(variance_adaptor_onnx, variance_adaptor_onnx)

    output, mel_lens = variance_adaptor.length_regulator(output, duration_rounded, model_config["max_seq_len"] * 2)
    mel_masks = get_mask_from_lengths(mel_lens, model_config["max_seq_len"] * 2)

    print("Starting export decoder ……")
    decoder = fastspeech2.decoder
    decoder.eval()
    decoder_input = (output, mel_masks)
    dec_output = decoder(*decoder_input)

    torch.onnx.export(decoder, decoder_input, os.path.join(args.output, "decoder.onnx"),
                      opset_version=11, do_constant_folding=True,
                      input_names=["output", "mel_masks"],
                      output_names=["dec_output"],
                      dynamic_axes={"output": {0: "batch_size", 1: "max_mel_len"},
                                    "mel_masks": {0: "batch_size", 1: "max_mel_len"}})

    print("Starting export postnet ……")
    postnet = Postnet(fastspeech2)
    postnet.eval()
    mel_output = postnet(dec_output)

    torch.onnx.export(postnet, dec_output, os.path.join(args.output, "postnet.onnx"),
                      opset_version=11, do_constant_folding=True,
                      input_names=["dec_output"],
                      output_names=["mel_output"],
                      dynamic_axes={"dec_output": {0: "batch_size", 1: "max_mel_len"}})

    print("Starting export vocoder ……")
    hifigan = Vocoder(vocoder, preprocess_config)
    hifigan.eval()
    wavs = hifigan(mel_output)

    torch.onnx.export(hifigan, mel_output, os.path.join(args.output, "hifigan.onnx"),
                      opset_version=11, do_constant_folding=True,
                      input_names=["mel_output"],
                      output_names=["wavs"],
                      dynamic_axes={"mel_output": {0: "batch_size", 1: "max_mel_len"}})
