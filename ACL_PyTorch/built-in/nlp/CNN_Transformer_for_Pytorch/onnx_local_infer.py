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
import torch
import argparse
import onnxruntime
import numpy as np
from jiwer import wer
from datasets import load_dataset
from transformers import Wav2Vec2Tokenizer
from glob import glob
from tqdm import tqdm
import time

if __name__ == '__main__':
    '''
    Using Example:

    python onnx_local_infer.py \
    --model_path=./models/wav2vec2-base-960h.onnx \
    --bin_file_path=./pre_data/validation \
    --pred_res_save_path=./onnx_local_infer_res/validation/transcriptions.txt \
    --which_dataset=validation

    python onnx_local_infer.py \
    --model_path=./models/wav2vec2-base-960h.onnx \
    --bin_file_path=./pre_data/clean \
    --pred_res_save_path=./onnx_local_infer_res/clean/transcriptions.txt \
    --which_dataset=clean

    python onnx_local_infer.py \
    --model_path=./models/wav2vec2-base-960h.onnx \
    --bin_file_path=./pre_data/other \
    --pred_res_save_path=./onnx_local_infer_res/other/transcriptions.txt \
    --which_dataset=other
    '''

    # 解析参数
    dataset_choices = ['validation', 'clean', 'other']
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--bin_file_path', required=True)
    parser.add_argument('--pred_res_save_path', required=True)
    parser.add_argument('--which_dataset', required=True, choices=dataset_choices)
    opt = parser.parse_args()
    model_path = opt.model_path
    bin_file_path = opt.bin_file_path
    pred_res_save_path = opt.pred_res_save_path
    which_dataset = opt.which_dataset

    print('[INFO] Current runing device is: {}'.format(onnxruntime.get_device()))

    # 加载分词器
    print('[INFO] Download or load tokenizer/model ...')
    tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')

    # 读取数据目录
    bin_file_list = glob(os.path.join(bin_file_path, '*.bin'))
    bin_file_num = len(bin_file_list)

    # 创建目录
    pardir = os.path.dirname(pred_res_save_path)
    if not os.path.exists(pardir):
        os.makedirs(pardir)

    # 推理
    print('[INFO] Infer on dataset ...')
    transcription_list = []
    total_infer_time = 0
    total_infer_num = 0

    with open(pred_res_save_path, 'wt', encoding='utf-8') as f_pred:
        onnx_run_sess = onnxruntime.InferenceSession(model_path)
        for i in tqdm(range(bin_file_num)):
            # 数据预处理
            input = np.fromfile(os.path.join(bin_file_path, str(i) + '.bin'), dtype=np.float32)
            input = input.reshape(1, len(input))

            # 推理
            start_time = time.perf_counter_ns()
            output = onnx_run_sess.run(None, {'input': input})
            end_time = time.perf_counter_ns()
            total_infer_time += end_time - start_time
            total_infer_num += 1

            # 后处理
            output = torch.from_numpy(output[0])
            predicted_ids = torch.argmax(output, dim=-1)
            transcription = tokenizer.batch_decode(predicted_ids)
            transcription_list.append(transcription[0])
            f_pred.write(transcription[0] + '\n')
    print('[INFO] Infer done!')

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

    # 推理时间
    print('[INFO] Time:')
    msg = 'total infer num: ' + str(total_infer_num) + '\n' + \
          'total infer time(ms): ' + str(total_infer_time / 1000 / 1000) + '\n' + \
          'average infer time(ms): ' + str(total_infer_time / total_infer_num / 1000 / 1000) + '\n'
    print(msg)
    with open(os.path.join(pardir, 'infer_time.txt'), 'wt', encoding='utf-8') as f_infer_time:
        f_infer_time.write(msg)
