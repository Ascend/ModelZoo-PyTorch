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

import os
import glob
import torch
import argparse
import numpy as np
from jiwer import wer
from datasets import load_dataset
from transformers import Wav2Vec2Tokenizer

if __name__ == '__main__':
    '''
    Using Example:

    python postprocess.py \
    --bin_file_path=./om_infer_res_validation \
    --res_save_path=./om_infer_res_validation/transcriptions.txt \
    --which_dataset=validation

    python postprocess.py \
    --bin_file_path=./om_infer_res_clean \
    --res_save_path=./om_infer_res_clean/transcriptions.txt \
    --which_dataset=clean

    python postprocess.py \
    --bin_file_path=./om_infer_res_other \
    --res_save_path=./om_infer_res_other/transcriptions.txt \
    --which_dataset=other
    '''

    dataset_choices = ['validation', 'clean', 'other']
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_file_path', required=True)
    parser.add_argument('--res_save_path', required=True)
    parser.add_argument('--which_dataset', required=True, choices=dataset_choices)
    opt = parser.parse_args()
    bin_file_path = opt.bin_file_path
    res_save_path = opt.res_save_path
    which_dataset = opt.which_dataset

    # 创建目录
    pardir = os.path.dirname(res_save_path)
    if not os.path.exists(pardir):
        os.makedirs(pardir)

    # 加载分词器
    print('[INFO] Download or load tokenizer/model ...')
    tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')

    # 后处理
    print('[INFO] Postprocessing ...')
    bin_file_list = glob.glob(os.path.join(bin_file_path, '*.bin'))
    bin_file_num = len(bin_file_list)
    transcription_list = []
    with open(res_save_path, 'wt', encoding='utf-8') as f_pred:
        for i in range(bin_file_num):
            output = np.fromfile(os.path.join(bin_file_path, str(i) + '.0.bin'), dtype=np.float32)
            output = output.reshape(1, len(output) // 32, 32)
            output = torch.from_numpy(output)
            predicted_ids = torch.argmax(output, dim=-1)
            transcription = tokenizer.batch_decode(predicted_ids)
            transcription_list.append(transcription[0])
            f_pred.write(transcription[0] + '\n')
    print('[INFO] Postprocess done!')

    #  计算WER
    print('[INFO] Now, calculate WER.')
    print('[INFO] Download or load dataset ...')
    if which_dataset == 'validation':
        librispeech_dataset = load_dataset('patrickvonplaten/librispeech_asr_dummy', 'clean', split='validation')
    elif which_dataset == 'clean':
        librispeech_dataset = load_dataset('librispeech_asr', 'clean', split='test')
    else:
        librispeech_dataset = load_dataset('librispeech_asr', 'other', split='test')

    wer_ = str(wer(librispeech_dataset['text'], transcription_list))
    print('[INFO] WER:', wer_)
    with open(os.path.join(pardir, 'wer.txt'), 'wt', encoding='utf-8') as f_wer:
        f_wer.write('WER: ' + wer_ + '\n')
