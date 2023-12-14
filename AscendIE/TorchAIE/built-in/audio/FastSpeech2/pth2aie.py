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
import torch_aie
import torch.nn as nn
import yaml
import numpy as np

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


class VarianceAdaptorSim(VarianceAdaptor):
    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptorSim, self).__init__(preprocess_config, model_config)

    def get_pitch_embedding(self, x, mask, control):
        prediction = self.pitch_predictor(x, mask)
        prediction = prediction * control
        embedding = self.pitch_embedding(torch.bucketize(prediction, self.pitch_bins))
        return prediction, embedding

    def get_energy_embedding(self, x, mask, control):
        prediction = self.energy_predictor(x, mask)
        prediction = prediction * control
        embedding = self.energy_embedding(torch.bucketize(prediction, self.energy_bins))
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
    parser.add_argument("--batch_size", type=int, default = 1, help="set batch size")
    parser.add_argument("--src_len", type=int, default = 250, help="set src_len")
    parser.add_argument("--aie_dir", type=str, default = "./aie_", help="save aie model to this dir")
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

    print("Starting trace encoder ……")
    encoder = Encoder(fastspeech2)
    encoder.eval()
    encoder_input = (texts, src_masks)
    enc_output = encoder(*encoder_input)
    encoder_traced_model = torch.jit.trace(encoder, encoder_input, strict = False)
    print("Trace encoder success.")
    encoder_input_info = [torch_aie.Input(min_shape = (args.batch_size, 1),
                                          max_shape = (args.batch_size, args.src_len), dtype = torch_aie.dtype.INT64),
                          torch_aie.Input(min_shape = (args.batch_size, 1),
                                          max_shape = (args.batch_size, args.src_len), dtype = torch_aie.dtype.BOOL)]
    print("start export encoder to aie ……")
    encoder_pt_model = torch_aie.compile(
        encoder_traced_model,
        inputs = encoder_input_info,
        precision_policy=torch_aie.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        soc_version="Ascend310P3",
        optimization_level=0
    )
    print("Export encoder to aie success.")
    encoder_pt_model.save("./output/" + args.aie_dir + "encoder.pt")
    print("Save encoder aie model success.")

    print("Starting trace variance_adaptor ……")
    variance_adaptor = fastspeech2.variance_adaptor
    variance_adaptor.eval()
    adaptor_input = (enc_output, src_masks, args.pitch_control, args.energy_control, args.duration_control)
    (output, duration_rounded) = variance_adaptor(*adaptor_input)
    trace_input = (enc_output, src_masks)
    variance_adaptor_traced_model = torch.jit.trace(variance_adaptor, trace_input, strict = False)
    print("Trace variance_adaptor success.")
    variance_adaptor_traced_model.save("./output/" + args.aie_dir + "variance_adaptor.pt")
    print("Save variance_adaptor aie model success.")

    output, mel_lens = variance_adaptor.length_regulator(output, duration_rounded, model_config["max_seq_len"] * 2)
    mel_masks = get_mask_from_lengths(mel_lens, model_config["max_seq_len"] * 2)

    print("Starting trace decoder ……")
    decoder = fastspeech2.decoder
    decoder.eval()
    decoder_input = (output, mel_masks)
    dec_output = decoder(*decoder_input)
    decoder_traced_model = torch.jit.trace(decoder, decoder_input, strict = False)
    print("Trace decoder success.")
    decoder_input_info = [torch_aie.Input(min_shape = (args.batch_size, 1, 256),
                                          max_shape = (args.batch_size, 2000, 256)),
                          torch_aie.Input(min_shape = (args.batch_size, 1),
                                          max_shape = (args.batch_size, 2000), dtype = torch_aie.dtype.BOOL)]
    print("start export decoder to aie ……")
    decoder_pt_model = torch_aie.compile(
        decoder_traced_model,
        inputs = decoder_input_info,
        precision_policy=torch_aie.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        soc_version="Ascend310P3",
        optimization_level=0
    )
    print("Export decoder to aie success.")
    decoder_pt_model.save("./output/" + args.aie_dir + "decoder.pt")
    print("Save decoder aie model success.")


    print("Starting trace postnet ……")
    postnet = Postnet(fastspeech2)
    postnet.eval()
    mel_output = postnet(dec_output)
    postnet_traced_model = torch.jit.trace(postnet, dec_output, strict = False)
    print("Trace postnet success.")
    postnet_input_info = [torch_aie.Input(min_shape = (args.batch_size, 1, 256),
                                          max_shape = (args.batch_size, 2000, 256))]
    print("start export postnet to aie ……")
    postnet_pt_model = torch_aie.compile(
        postnet_traced_model,
        inputs = postnet_input_info,
        precision_policy=torch_aie.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        soc_version="Ascend310P3",
        optimization_level=0
    )
    print("Export postnet to aie success.")
    postnet_pt_model.save("./output/" + args.aie_dir + "postnet.pt")
    print("Save postnet aie model success.")

    print("Starting trace hifigan ……")
    hifigan = Vocoder(vocoder, preprocess_config)
    hifigan.eval()
    hifigan_traced_model = torch.jit.trace(hifigan, mel_output, strict = False)
    print("Trace hifigan success.")
    vocoder_input_info_1 = [torch_aie.Input((args.batch_size, 250, 80))]
    vocoder_input_info_2 = [torch_aie.Input((args.batch_size, 500, 80))]
    vocoder_input_info_3 = [torch_aie.Input((args.batch_size, 750, 80))]
    vocoder_input_info_4 = [torch_aie.Input((args.batch_size, 1000, 80))]
    vocoder_input_info_5 = [torch_aie.Input((args.batch_size, 1250, 80))]
    vocoder_input_info_6 = [torch_aie.Input((args.batch_size, 1500, 80))]
    vocoder_input_info_7 = [torch_aie.Input((args.batch_size, 1750, 80))]
    vocoder_input_info_8 = [torch_aie.Input((args.batch_size, 2000, 80))]
    vocoder_input_info = [vocoder_input_info_1, vocoder_input_info_2, vocoder_input_info_3, vocoder_input_info_4,
                          vocoder_input_info_5, vocoder_input_info_6, vocoder_input_info_7, vocoder_input_info_8]
    print("start export hifigan to aie ……")
    hifigan_pt_model = torch_aie.compile(
        hifigan_traced_model,
        inputs = vocoder_input_info,
        precision_policy=torch_aie.PrecisionPolicy.FP16,
        allow_tensor_replace_int=True,
        soc_version="Ascend310P3",
        optimization_level=0
    )
    print("Export hifigan to aie success.")
    hifigan_pt_model.save("./output/" + args.aie_dir + "hifigan.pt")
    print("Save hifigan aie model success.")
