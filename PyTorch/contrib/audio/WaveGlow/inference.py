# Copyright (c) Soumith Chintala 2016,
# All rights reserved
#
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://spdx.org/licenses/BSD-3-Clause.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from scipy.io.wavfile import write
import torch
from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser


def main(mel_files, waveglow_path, sigma, output_dir, sampling_rate, is_fp16,
         denoiser_strength):
    mel_files = files_to_list(mel_files)
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    torch.npu.set_device(0)
    waveglow.npu().eval()
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).npu()

    for i, file_path in enumerate(mel_files):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        mel = torch.load(file_path)
        mel = torch.autograd.Variable(mel.npu())
        mel = torch.unsqueeze(mel, 0)
        mel = mel.half() if is_fp16 else mel
        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=sigma)
            if denoiser_strength > 0:
                audio = denoiser(audio, denoiser_strength)
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = os.path.join(
            output_dir, "{}_synthesis.wav".format(file_name))
        write(audio_path, sampling_rate, audio)
        print(audio_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    main(args.filelist_path, args.waveglow_path, args.sigma, args.output_dir,
         args.sampling_rate, args.is_fp16, args.denoiser_strength)
