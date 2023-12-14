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


import re
import os
import stat
import argparse
import time
from tqdm import tqdm

import torch
import torch_aie
from torch.utils.data import DataLoader
import yaml
import numpy as np
from scipy.io import wavfile

from utils.tools import to_device, get_mask_from_lengths, pad
from dataset import TextDataset


def expand(batch, predicted):
    out = list()

    for i, vec in enumerate(batch):
        expand_size = predicted[i].item()
        out.append(vec.expand(max(int(expand_size), 0), -1))
    out = torch.cat(out, 0)

    return out


def LR(x, duration, max_len=None):
    output = list()
    mel_len = list()
    for batch, expand_target in zip(x, duration):
        expanded = expand(batch, expand_target)
        output.append(expanded)
        mel_len.append(expanded.shape[0])

    if max_len is not None:
        output = pad(output, max_len)
    else:
        output = pad(output)

    return output, torch.LongTensor(mel_len)


def fastspeech2_infer(texts, src_masks, fastspeech2_aie, control_values):
    encoder_aie, variance_adaptor_aie, decoder_aie, postnet_aie = fastspeech2_aie
    pitch_control, energy_control, duration_control = np.array(control_values)

    # encoder infer
    enc_output = encoder_aie(texts.to("npu"), src_masks.to("npu"))

    # variance_adaptor infer
    output_tmp = variance_adaptor_aie(enc_output.to("cpu")[0].unsqueeze(0), src_masks.to("cpu"))

    output = output_tmp[0]
    d_rounded = output_tmp[1]
    output, mel_lens = LR(output, d_rounded)
    mel_masks = get_mask_from_lengths(mel_lens)

    # decoder infer
    dec_output = decoder_aie(output.to("npu"), mel_masks.to("npu")).to("cpu")

    # postnet infer
    mel_output = postnet_aie(dec_output.to("npu"))

    return (mel_output.to("cpu"), mel_lens)


def get_shape(wav_shape):
    multi_shape = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    for shape in multi_shape:
        if wav_shape < shape:
            return shape
    return max(multi_shape)


def hifigan_infer(ids, mel_output, mel_lens, preprocess_config, train_config, hifigan_om):
    mel_output = mel_output.unsqueeze(0)
    model_shape = get_shape(mel_output[0].shape[1])
    x_pad = np.pad(mel_output[0], ((0, 0), (0, model_shape - mel_output[0].shape[1]), (0, 0)),
                   mode="constant", constant_values=0)

    wavs = hifigan_om(torch.from_numpy(x_pad).to("npu"))
    wavs = wavs.to("cpu").numpy()

    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    lengths = mel_lens * preprocess_config["preprocessing"]["stft"]["hop_length"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    wavs = [wav for wav in wavs]

    for i in range(len(mel_lens)):
        wavs[i] = wavs[i][: lengths[i]]

    for wav, id in zip(wavs, ids):
        wavfile.write(os.path.join(train_config["path"]["result_path"], f"{id}.wav"), sampling_rate, wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None,
                        help="path to a source file with format like train.txt and val.txt")
    parser.add_argument("--aie_path", type=str, default="output/om", help="path to load FastSpeech2 aie model")
    parser.add_argument("--encoder", type=str, default="encoderencoder.pt", help="path to aie")
    parser.add_argument("--variance_adaptor", type=str, default="variance_adaptor.pt", help="path to aie")
    parser.add_argument("--decoder", type=str, default="decoder.pt", help="path to aie")
    parser.add_argument("--postnet", type=str, default="postnet.pt", help="path to aie")
    parser.add_argument("--vocoder", type=str, default="hifigan.pt", help="path to aie")
    parser.add_argument("-p", "--preprocess_config", type=str, help="path to preprocess.yaml")
    parser.add_argument("-t", "--train_config", type=str, help="path to train.yaml")
    parser.add_argument("-vp", "--pitch_control", type=float, default=1.0,
                        help="control the pitch of the whole utterance, larger value for higher pitch")
    parser.add_argument("-ve", "--energy_control", type=float, default=1.0,
                        help="control the energy of the whole utterance, larger value for larger volume")
    parser.add_argument("-vd", "--duration_control", type=float, default=1.0,
                        help="control the speed of the whole utterance, larger value for slower speaking rate")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    args = parser.parse_args()
    torch_aie.set_device(args.device_id)

    # read config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)

    # load dataset
    dataset = TextDataset(args.source, preprocess_config)
    batchs = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    print("load dataset success")

    # load model
    encoder_aie = torch.jit.load("./output/aie_encoder.pt")
    variance_adaptor_aie = torch.jit.load("./output/aie_variance_adaptor.pt")
    decoder_aie = torch.jit.load("./output/aie_decoder.pt")
    postnet_aie = torch.jit.load("./output/aie_postnet.pt")
    fastspeech2_aie = (encoder_aie, variance_adaptor_aie, decoder_aie, postnet_aie)
    hifigan_aie = torch.jit.load("./output/aie_hifigan.pt")
    print("load aie model success")

    control_values = args.pitch_control, args.energy_control, args.duration_control

    start_time = time.time()
    cnt = 0
    batch_data = []
    for data in batchs:
        data = to_device(data, 'cpu')
        batch_data.append(data)
    batch_data.sort(key=lambda x : x[5])
    for batch in tqdm(batch_data):
        cnt += 1
        batch = to_device(batch, 'cpu')
        src_masks = get_mask_from_lengths(batch[4], batch[5])
        mel_output, mel_lens = fastspeech2_infer(batch[3], src_masks, fastspeech2_aie, control_values)
        hifigan_infer(batch[0], mel_output, mel_lens, preprocess_config, train_config, hifigan_aie)
    end_time = time.time()
    fps = (cnt * args.batch_size) / (end_time - start_time)
    res = 'fps: {}'.format(fps)
    print(res)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open('result.txt', flags, modes), 'w') as f:
        f.write(res)
