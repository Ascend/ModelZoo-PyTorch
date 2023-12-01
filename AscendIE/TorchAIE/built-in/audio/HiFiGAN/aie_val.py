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
import argparse
import json
from env import AttrDict

import torch
import torch.nn.functional as F
import torch_aie
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm

from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav


class BatchDataLoader:
    def __init__(self, input_wavs_dir, config, batch_size):
        self.input_wavs_dir = input_wavs_dir
        self.filelist = os.listdir(args.input_wavs_dir)
        self.sample_num = len(self.filelist)
        self.cfg = config
        self.batch_size = batch_size

    def __len__(self):
        return self.sample_num // self.batch_size + int(self.sample_num % self.batch_size > 0)

    @staticmethod
    def get_mel(x, cfg):
        return mel_spectrogram(x, cfg.n_fft, cfg.num_mels, cfg.sampling_rate,
                               cfg.hop_size, cfg.win_size, cfg.fmin, cfg.fmax)

    @staticmethod
    def get_shape(wav_shape):
        multi_shape = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
        for shape in multi_shape:
            if wav_shape < shape:
                return shape
        return max(multi_shape)

    def __getitem__(self, item):
        if (item + 1) * self.batch_size <= self.sample_num:
            slice_end = (item + 1) * self.batch_size
        else:
            slice_end = self.sample_num

        mel_specs = []
        mel_lens = []
        wav_names = []
        for path in self.filelist[item * self.batch_size:slice_end]:
            wav, sr = load_wav(os.path.join(self.input_wavs_dir, path))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav)
            mel_spec = self.get_mel(wav.unsqueeze(0), self.cfg)
            mel_spec = mel_spec.unsqueeze(2)
            mel_specs.append(mel_spec)
            mel_lens.append(mel_spec.shape[3])
            wav_names.append(path)
        max_len = max(mel_lens)

        mel_specs_pad = []
        for mel_spec in mel_specs:
            model_shape = self.get_shape(max_len)
            mel_spec_pad = F.pad(mel_spec, (0, model_shape - mel_spec.shape[3], 0, 0, 0, 0, 0, 0), "constant", 0)
            mel_specs_pad.append(mel_spec_pad)
        mel_specs = torch.cat(mel_specs_pad)

        return (mel_specs, mel_lens, wav_names)


def inference(model, dataloader, cfg):
    for i in tqdm(range(len(dataloader))):
        mel_specs_pad, mel_lens, wav_names = dataloader[i]
        data_len = len(mel_lens)
        if i == len(dataloader) - 1:
            mel_specs_pad = F.pad(mel_specs_pad, (0, 0, 0, 0, 0, 0, 0, args.batch_size-data_len), "constant", 0)
            mel_lens.extend([0]*(args.batch_size-data_len))
            wav_names.extend(['']*(args.batch_size-data_len))
        inputs = torch.tensor([mel_specs_pad.numpy()])
        if inputs.dim() == 5 and inputs.size(0) == 1:
            inputs = inputs.squeeze(0)
        inputs_npu = inputs.to("npu:0")
        output = model(inputs_npu)
        wavs = output.to("cpu").numpy()
        if i == len(dataloader) - 1:
            wavs = wavs[:data_len]
            mel_lens = mel_lens[:data_len]
            wav_names = wav_names[:data_len]
        wavs = (wavs * MAX_WAV_VALUE).astype('int16')

        for i in range(len(mel_lens)):
            wav = wavs[i].squeeze()[: mel_lens[i] * 256]
            output_file = os.path.join(args.output_wavs_dir, wav_names[i])
            write(output_file, cfg.sampling_rate, wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', type=str, default='LJSpeech-1.1/wavs')
    parser.add_argument('--output_wavs_dir', type=str, default='output/gen_wavs')
    parser.add_argument('--aie_dir', type=str, default='./aie_model.ts')
    parser.add_argument('--config_file', type=str, default='config.json')
    parser.add_argument('--batch-size', type=int, default=1, help='om batch size')
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    torch_aie.set_device(0)

    # load config
    with open(args.config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    config = AttrDict(json_config)

    # load model
    model = torch.jit.load(args.aie_dir)
    print("load model success")

    # load dataset
    dataloader = BatchDataLoader(args.input_wavs_dir, config, args.batch_size)

    # infer om
    inference(model, dataloader, config)
