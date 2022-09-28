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
# ============================================================================

import csv
import math
import os
import numpy as np
import torch
import argparse
from tqdm import tqdm


#============================================================================
# Functions
#============================================================================
def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def run_preprocess(args):
    input_matched = os.path.join(args.datasets_path, "dev_matched.tsv")
    input_mismatched = os.path.join(args.datasets_path, "dev_mismatched.tsv")
    assert os.path.exists(input_matched), f"{input_matched} doesn't exists"
    assert os.path.exists(input_mismatched), f"{input_mismatched} doesn't exists"
    match_data = _read_tsv(input_matched)
    mismatch_data = _read_tsv(input_mismatched)

    match = list((l[8], l[9]) for l in match_data[1:])
    mismatch = list((l[8], l[9]) for l in mismatch_data[1:])

    from transformers import AutoTokenizer
    tokenizers = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    input_match = tokenizers(match, padding=True, truncation=True, max_length=256, return_tensors="pt")
    input_mismatch = tokenizers(mismatch, padding=True, truncation=True, max_length=256, return_tensors="pt")

    input_match_ids = torch.tensor(input_match["input_ids"], dtype=torch.int32)
    input_match_mask = torch.tensor(input_match["attention_mask"], dtype=torch.int32)
    pad_match = torch.nn.ZeroPad2d(padding=(0, 256 - input_match_ids.shape[1], 0, 0))
    input_match_ids = pad_match(input_match_ids)
    input_match_mask = pad_match(input_match_mask)

    input_mismatch_ids = torch.tensor(input_mismatch["input_ids"], dtype=torch.int32)
    input_mismatch_mask = torch.tensor(input_mismatch["attention_mask"], dtype=torch.int32)
    pad_mismatch = torch.nn.ZeroPad2d(padding=(0, 256 - input_mismatch_ids.shape[1], 0, 0))
    input_mismatch_ids = pad_mismatch(input_mismatch_ids)
    input_mismatch_mask = pad_mismatch(input_mismatch_mask)

    batch_match = math.ceil(input_match_ids.shape[0] / args.batch_size)
    batch_mismatch = math.ceil(input_mismatch_ids.shape[0] / args.batch_size)

    input_match_ids_list = []
    input_match_mask_list = []
    input_mismatch_ids_list = []
    input_mismatch_mask_list = []
    for i in tqdm(range(batch_match), desc="generate match bin data"):
        if i < batch_match - 1:
            input_match_ids_list.append(input_match_ids[i * args.batch_size:(i + 1) * args.batch_size, :])
            input_match_mask_list.append(input_match_mask[i * args.batch_size:(i + 1) * args.batch_size, :])
        else:
            tmp_ids = input_match_ids[i * args.batch_size:input_match_ids.shape[0], :]
            tmp_mask = input_match_mask[i * args.batch_size:input_match_mask.shape[0], :]
            tmp = torch.zeros(args.batch_size - tmp_ids.shape[0], 256)
            input_match_ids_list.append(torch.cat((tmp_ids, tmp), 0))
            input_match_mask_list.append(torch.cat((tmp_mask, tmp), 0))

    for i in tqdm(range(batch_mismatch), desc="generate mismatch bin data"):
        if i < batch_mismatch - 1:
            input_mismatch_ids_list.append(input_mismatch_ids[i * args.batch_size:(i + 1) * args.batch_size, :])
            input_mismatch_mask_list.append(input_mismatch_mask[i * args.batch_size:(i + 1) * args.batch_size, :])
        else:
            tmp_ids = input_mismatch_ids[i * args.batch_size:input_mismatch_ids.shape[0], :]
            tmp_mask = input_mismatch_mask[i * args.batch_size:input_mismatch_mask.shape[0], :]
            tmp = torch.zeros(args.batch_size - tmp_ids.shape[0], 256)
            input_mismatch_ids_list.append(torch.cat((tmp_ids, tmp), 0))
            input_mismatch_mask_list.append(torch.cat((tmp_mask, tmp), 0))

    for i in tqdm(range(batch_match), desc="write match data to bin"):
        ids =  input_match_ids_list[i]
        mask = input_match_mask_list[i]
        input_match_ids_file_path = os.path.join(args.pre_data_save_path, 'match', 'input_ids', 'input_{}.bin'.format(i))
        input_match_mask_file_path = os.path.join(args.pre_data_save_path, 'match', 'input_mask', 'input_{}.bin'.format(i))
        if not os.path.exists(os.path.join(args.pre_data_save_path, 'match', 'input_ids')):
            os.makedirs(os.path.join(args.pre_data_save_path, 'match', 'input_ids'))
        if not os.path.exists(os.path.join(args.pre_data_save_path, 'match', 'input_mask')):
            os.makedirs(os.path.join(args.pre_data_save_path, 'match', 'input_mask'))
        if not os.path.exists(input_match_ids_file_path):
            os.system(r"touch {}".format(input_match_ids_file_path))
        if not os.path.exists(input_match_mask_file_path):
            os.system(r"touch {}".format(input_match_mask_file_path))
        (np.asarray(ids, dtype=np.int32)).tofile(input_match_ids_file_path)
        (np.asarray(mask, dtype=np.int32)).tofile(input_match_mask_file_path)
    
    for i in tqdm(range(batch_mismatch), desc="write mismatch data to bin"):
        mis_ids = input_mismatch_ids_list[i]
        mis_mask = input_mismatch_mask_list[i]
        input_mismatch_ids_file_path = os.path.join(args.pre_data_save_path, 'mismatch', 'input_ids', 'input_{}.bin'.format(i))
        input_mismatch_mask_file_path = os.path.join(args.pre_data_save_path, 'mismatch', 'input_mask', 'input_{}.bin'.format(i))
        if not os.path.exists(os.path.join(args.pre_data_save_path, 'mismatch', 'input_ids')):
            os.makedirs(os.path.join(args.pre_data_save_path, 'mismatch', 'input_ids'))
        if not os.path.exists(os.path.join(args.pre_data_save_path, 'mismatch', 'input_mask')):
            os.makedirs(os.path.join(args.pre_data_save_path, 'mismatch', 'input_mask'))
        if not os.path.exists(input_mismatch_ids_file_path):
            os.system(r"touch {}".format(input_mismatch_ids_file_path))
        if not os.path.exists(input_mismatch_mask_file_path):
            os.system(r"touch {}".format(input_mismatch_mask_file_path))
        (np.asarray(mis_ids, dtype=np.int32)).tofile(input_mismatch_ids_file_path)
        (np.asarray(mis_mask, dtype=np.int32)).tofile(input_mismatch_mask_file_path)


#============================================================================
# Main
#============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path', default='./MNLI/')
    parser.add_argument('--pre_data_save_path', default='./pre_data/')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    run_preprocess(args)
