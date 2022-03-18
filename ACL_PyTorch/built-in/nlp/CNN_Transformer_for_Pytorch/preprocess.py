# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

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
            bin_file_path = os.path.join(pre_data_save_path, str(count) + '.bin')
            speech.tofile(bin_file_path)
            f_info.write(
                str(count) + ' ' + os.path.join(pre_data_save_path,
                                                str(count) + '.bin ') + str(speech.shape).replace(' ', '') + '\n')
    print('[INFO] Dataset preprocess done!')

    # 记录所有padding长度
    padding_len_list = sorted(padding_len_set)
    with open(os.path.join(pre_data_save_path, 'padding_lens.txt'), 'wt', encoding='utf-8') as f_padding_len:
        for item in padding_len_list:
            f_padding_len.write(str(item) + '\n')
