
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
import sys
import argparse
import math
import numpy as np
from scipy.io.wavfile import write
import torch

MAX_WAV_VALUE = 32768.0


def main(file_dir, output_dir, sampling_rate):
    bin_list = []
    if os.path.exists(file_dir):
        sec_dirs = os.listdir(file_dir)
        for sec_dir in sec_dirs:
            file_name = os.path.join(file_dir, sec_dir)
            file_type = os.path.splitext(file_name)[1]
            if file_type in ['.bin']:
                bin_list.append(file_name)
                audio = np.fromfile(file_name, dtype=np.float32)
                with torch.no_grad():
                    audio = audio * MAX_WAV_VALUE
                audio = audio.squeeze()
                audio = audio.astype('int16')
                audio_path = os.path.join(
                    output_dir, "{}_syn.wav".format(os.path.basename(os.path.splitext(file_name)[0])))
                write(audio_path, sampling_rate, audio)
                print(audio_path)
    else:
        raise ValueError("Dir {} not exists!".format(file_dir))


if __name__ == "__main__":
    '''
    Example:
        python3.7 Waveglow_postprocess.py \
            -f ./result/LJLJ001-0001_0.bin
            -o ./syn_wavs
    '''
    import argparse

    parser = argparse.ArgumentParser(description='Postprocess of ais_infer result')
    parser.add_argument('-f', "--file_dir", required=True)
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("--sampling_rate", default=22050, type=int)

    args = parser.parse_args()

    main(args.file_dir, args.output_dir, args.sampling_rate)

