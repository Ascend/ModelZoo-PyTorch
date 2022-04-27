# Copyright 2018 NVIDIA Corporation. All Rights Reserved.
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
# ============================================================================
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
import argparse
import time
import math
import numpy as np
from tqdm import tqdm
import librosa
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from acl_net import Net


def genSpoof_list(dir_meta):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        key = line.strip().split(' ')[1]
        file_list.append(key)
    return file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_ASVspoof2019_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key)'''

        self.list_IDs = list_IDs
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + key + '.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key


def get_parser():
    parser = argparse.ArgumentParser(description='RawNet2')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--device_id', type=int, default=1,
                        help='device id')
    parser.add_argument('--om_path', type=str, default="rawnet2.om",
                        help='path to the om model')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/ASVspoof_database/',
                        help='Change this to user\'s full directory address of LA database.')
    '''
    % database_path/
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''
    parser.add_argument('--protocols_path', type=str, default='/your/path/to/protocols/ASVspoof_database/',
                        help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt 
    '''
    return parser


if __name__ == "__main__":
    '''
    Example:
        python3.7 om_infer.py \
            --batch_size=1 \
            --om_path=rawnet2_modify.om \
            --eval_output='rawnet2_modify_om.txt' \
            --database_path='data/LA/' \
            --protocols_path='data/LA/'
    '''
    parser = get_parser()
    args = parser.parse_args()

    # Load dataset
    protocal_dir = os.path.join(args.protocols_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
    file_eval = genSpoof_list(protocal_dir)
    database_dir = os.path.join(args.database_path + 'ASVspoof2019_LA_eval/')
    eval_set = Dataset_ASVspoof2019_eval(list_IDs=file_eval, base_dir=database_dir)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = Net(device_id=args.device_id, model_path=args.om_path)

    # Evaluation for RawNet2 om model
    with open(args.eval_output, 'w+') as fh:
        for idx, (batch_x, utt_id) in tqdm(enumerate(eval_loader)):
            fname_list = []
            score_list = []
            n, d = batch_x.shape
            if n != args.batch_size:
                m = (0, 0, 0, args.batch_size - n)
                batch_x = F.pad(batch_x, m, 'constant', 0)
            batch_x = batch_x.numpy().astype(np.float32)
            batch_out = model(batch_x)
            batch_out = torch.from_numpy(np.array(batch_out).astype(np.float32))
            batch_score = (batch_out[:, :, 1]).data.cpu().numpy().ravel()

            # add outputs
            if len(batch_score) != len(utt_id):
                batch_score = batch_score[:len(utt_id)]
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
    print('Scores saved to {}'.format(args.eval_output))
    
    del model