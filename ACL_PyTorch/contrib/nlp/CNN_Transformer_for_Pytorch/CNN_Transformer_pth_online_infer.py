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

import numpy as np
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch
import argparse
import os
from tqdm import tqdm
from jiwer import wer
import time

if __name__ == '__main__':
    '''
    Using Example:

    python pth_online_infer.py \
    --pred_res_save_path=./pth_online_infer_res/validation/transcriptions.txt \
    --which_dataset=validation

    python pth_online_infer.py \
    --pred_res_save_path=./pth_online_infer_res/clean/transcriptions.txt \
    --which_dataset=clean

    python pth_online_infer.py \
    --pred_res_save_path=./pth_online_infer_res/other/transcriptions.txt \
    --which_dataset=other
    '''

    # 解析参数
    dataset_choices = ['validation', 'clean', 'other']
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_res_save_path', required=True)
    parser.add_argument('--which_dataset', required=True, choices=dataset_choices)
    opt = parser.parse_args()
    pred_res_save_path = opt.pred_res_save_path
    which_dataset = opt.which_dataset

    # 加载分词器和模型
    print('[INFO] Download or load tokenizer/model ...')
    tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

    # 加载数据集
    print('[INFO] Download or load dataset ...')
    if which_dataset == 'validation':
        librispeech_dataset = load_dataset('patrickvonplaten/librispeech_asr_dummy', 'clean', split='validation')
    elif which_dataset == 'clean':
        librispeech_dataset = load_dataset('librispeech_asr', 'clean', split='test')
    else:
        librispeech_dataset = load_dataset('librispeech_asr', 'other', split='test')

    # 创建目录
    pardir = os.path.dirname(pred_res_save_path)
    if not os.path.exists(pardir):
        os.makedirs(pardir)

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

    # 推理
    print('[INFO] Infer on dataset ...')
    transcription_list = []
    total_infer_time = 0
    total_infer_num = 0
    with open(pred_res_save_path, 'wt', encoding='utf-8') as f_pred:
        for count, item in enumerate(tqdm(librispeech_dataset)):
            # 数据预处理
            speech, _ = sf.read(item['file'])
            len_speech = len(speech)
            padding_len = decice_padding_len(len_speech)
            speech = np.pad(speech, (0, padding_len - len_speech), 'constant',
                            constant_values=(0, 0)).astype(np.float32).reshape(1, padding_len)

            # 推理
            speech = torch.from_numpy(speech)
            start_time = time.perf_counter_ns()
            output = model(speech)
            end_time = time.perf_counter_ns()
            total_infer_time += end_time - start_time
            total_infer_num += 1

            # 后处理
            predicted_ids = torch.argmax(output.logits, dim=-1)
            transcription = tokenizer.batch_decode(predicted_ids)
            transcription_list.append(transcription[0])
            f_pred.write(transcription[0] + '\n')
    print('[INFO] Infer done!')

    #  计算WER
    print('[INFO] Now, calculate WER.')
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
