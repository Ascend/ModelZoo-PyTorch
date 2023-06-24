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

import os
import sys
import json

from tqdm import tqdm
import torchaudio
import torch.nn.functional as F

from speechbrain.pretrained import EncoderClassifier
sys.path.insert(0, 'speechbrain_onnx')
from templates.speaker_id.mini_librispeech_prepare import prepare_mini_librispeech

prepare_mini_librispeech(data_folder='data', save_json_train='train.json', save_json_valid='valid.json',
                         save_json_test='test.json', split_ratio=[0, 0, 100])

if not os.path.exists('mini_librispeech_test_bin'):
    os.makedirs('mini_librispeech_test_bin')

file = open('mini_librispeech_test.info', 'w')
classifier = EncoderClassifier.from_hparams(source='best_model', savedir='best_model')

with open('test.json', 'r') as f:
    data_info = json.load(f)
    
    for i, (key, value) in enumerate(tqdm(data_info.items())):
        wav_file = 'data' + value['wav'][11:] # prefix length 11
        signal, fs = torchaudio.load(wav_file)
        feats = classifier.extract_feats(signal)
        # pad signal
        pad = (feats.shape[1] // 100 + 1) * 100 - feats.shape[1]
        feats = F.pad(feats, (0, 0, 0, pad, 0, 0), value=0)

        # dump bin file
        output = value['wav'].split('/')[-1][:-4]
        output = f'mini_librispeech_test_bin/{output}bin'
        feats.numpy().tofile(output)
        # write shape info
        file.write(f'{str(i)} {output} ({str(feats.shape[0])},{str(feats.shape[1])},{str(feats.shape[2])})')
        file.write('\n')

    print('data preprocess done')
    file.close()