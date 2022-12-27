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

from transformers import Wav2Vec2Tokenizer
from datasets import load_dataset
import soundfile as sf
import numpy as np
import os
import argparse
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    '''
    Using Example:

    python preprocess.py --pre_data_save_path=./pre_data/validation --which_dataset=validation

    python preprocess.py --pre_data_save_path=./pre_data/clean --which_dataset=clean

    python preprocess.py --pre_data_save_path=./pre_data/other --which_dataset=other
    '''

    # 解析参数
    dataset_choices = ['validation', 'clean', 'other']
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_data_save_path', required=True)
    parser.add_argument('--which_dataset', required=True, choices=dataset_choices)
    opt = parser.parse_args()
    pre_data_save_path = opt.pre_data_save_path
    which_dataset = opt.which_dataset

    # 加载分词器
    print('[INFO] Download or load tokenizer/model ...')
    tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')

    # 加载数据集
    print('[INFO] Download or load dataset ...')
    if which_dataset == 'validation':
        librispeech_dataset = load_dataset('patrickvonplaten/librispeech_asr_dummy', 'clean', split='validation')
    elif which_dataset == 'clean':
        librispeech_dataset = load_dataset('librispeech_asr', 'clean', split='test')
    else:
        librispeech_dataset = load_dataset('librispeech_asr', 'other', split='test')

    # 创建目录
    if os.path.exists(pre_data_save_path):
        shutil.rmtree(pre_data_save_path)
    os.makedirs(pre_data_save_path)

    # 决定数据补齐长度
    def decice_padding_len(org_len):
        if org_len <= 10000:
            return 10000
        else:
            res = divmod(org_len, 10000)
            if res[1] == 0:
                return res[0] * 10000
            else:
                return (res[0] + 1) * 10000

    # 处理数据集
    print('[INFO] Preprocess the dataset ...')
    padding_len_set = set()
    info_file_path = os.path.join(opt.pre_data_save_path, 'bin_file.info')
    with open(info_file_path, 'wt', encoding='utf-8') as f_info:
        for count, item in enumerate(tqdm(librispeech_dataset)):
            speech, _ = sf.read(item['file'])
            len_speech = len(speech)
            padding_len = decice_padding_len(len_speech)
            padding_len_set.add(padding_len)
            speech = np.pad(speech, (0, padding_len - len_speech), 'constant',
                            constant_values=(0, 0)).astype(np.float32).reshape(1, padding_len)
            npy_file_path = os.path.join(pre_data_save_path, str(count) + '.npy')
            np.save(npy_file_path, speech)
            f_info.write(
                str(count) + ' ' + os.path.join(pre_data_save_path,
                                                str(count) + '.bin ') + str(speech.shape).replace(' ', '') + '\n')
    print('[INFO] Dataset preprocess done!')

    # 记录所有padding长度
    padding_len_list = sorted(padding_len_set)
    with open(os.path.join(pre_data_save_path, 'padding_lens.txt'), 'wt', encoding='utf-8') as f_padding_len:
        for item in padding_len_list:
            f_padding_len.write(str(item) + '\n')
