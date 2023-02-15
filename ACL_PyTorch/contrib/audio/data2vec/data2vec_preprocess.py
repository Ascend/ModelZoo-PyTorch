# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
from tqdm import tqdm
import json
import numpy as np
import scipy.signal as sps
import soundfile
import argparse


input_size = 559280
new_rate = 16000
AUDIO_MAXLEN = input_size


def normalize(x):
    """You must call this before padding.
    Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/wav2vec2/processor.py#L101
    Fork TF to numpy
    """
    # -> (1, seqlen)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return np.squeeze((x - mean) / np.sqrt(var + 1e-5))


def pad_speech(data, max_length=AUDIO_MAXLEN):
    '''padding data after feature extraction'''

    # Truncate data that exceeds the maximum length
    if data.shape[1] > max_length:
        data = data[:, 0:max_length]
    else:
        # Use 0 for padding to max_length at axis 0.
        data = np.pad(
            data, ((0, 0), (0, max_length-data.shape[1])), 'constant')
    return data


def load_speech(speech_path):
    data, sampling_rate = soundfile.read(speech_path)
    if (len(data) > AUDIO_MAXLEN):
        return None
    samples = round(len(data) * float(new_rate) / sampling_rate)
    new_data = sps.resample(data, samples)
    speech = np.array(new_data, dtype=np.float32)
    speech = normalize(speech)[None]
    speech = pad_speech(speech).astype(np.float32)
    return speech


def find_files(path, pattern="*.flac"):
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{pattern}', recursive=True):
       filenames.append(filename)
    return filenames


def pre_process(speech_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename_map_path = "./data/filename_map.json"
    filelist = find_files(speech_dir, pattern="*.flac")
    filename_map = dict()
    sample_count = 0
    data_count = 0
    for file in tqdm(filelist):
        data = load_speech(file)
        if data is None:
            continue
        sample_count += 1
        key = str(data_count)
        filename_map[key] = os.path.basename(file).replace('.flac', '')
        data_file_name = os.path.join(output_dir, key+".bin")
        data_count += 1
        np.array(data).tofile(data_file_name)

    with open(filename_map_path, 'w', encoding='utf-8') as f:
        json.dump(filename_map, f)
    print("Binary file saved in: ", output_dir)

    # 把原始多个真实文本文件写到一个文件
    ground_truth_texts_paths = find_files(speech_dir, pattern="*.txt")
    lines = []
    for file in ground_truth_texts_paths:
        with open(file, mode='r', encoding='utf-8') as fread:
            lines += fread.readlines()

    ground_truth_file_path = os.path.join("./data", "ground_truth_texts.txt")
    with open(ground_truth_file_path, mode='w', encoding='utf-8') as fwrite:
        for line in lines:
            fwrite.write(line)
    print("Ground truth texts file saved in: ", ground_truth_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='dataset path',
                        default="/opt/npu/LibriSpeech/test-clean/")
    parser.add_argument('--output', help='out bin path',
                        default="data/bin_out/")
    args = parser.parse_args()

    base_dir = args.input
    out_dir = args.output

    pre_process(base_dir, out_dir)

