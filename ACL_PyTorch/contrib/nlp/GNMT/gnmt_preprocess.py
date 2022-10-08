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
import torch
import numpy as np
from tqdm import tqdm
sys.path.append('./DeepLearningExamples/PyTorch/Translation/GNMT/')
from seq2seq.data import config
from seq2seq.data.dataset import RawTextDataset
from seq2seq.data.tokenizer import Tokenizer

def run_preprocess(args):
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    tokenizer = Tokenizer()
    tokenizer.set_state(checkpoint['tokenizer'])

    if os.path.exists(args.pre_data_save_path) == False:
        os.mkdir(args.pre_data_save_path)
    f_test_de = open(os.path.join(args.data_path, 'newstest2014.de'), "rt")
    f_test_en = open(os.path.join(args.data_path, 'newstest2014.en'), "rt")
    f_test_en_len = open(os.path.join(args.pre_data_save_path, "test_en.txt"), "wt")
    f_test_de_len = open(os.path.join(args.pre_data_save_path, "test_de.txt"), "wt")

    data = RawTextDataset(raw_datafile=os.path.join(args.data_path, 'newstest2014.en'), tokenizer=tokenizer)
    loader = data.get_loader(
    batch_size=1,
    batch_first=True,
    pad=True)

    bos = torch.tensor([[config.BOS]], dtype=torch.int32)
    bos_np = bos.view(-1, 1).numpy()

    count = 0

    if os.path.exists(os.path.join(args.pre_data_save_path, "input_encoder")) == False:
        os.mkdir(os.path.join(args.pre_data_save_path, "input_encoder"))
    if os.path.exists(os.path.join(args.pre_data_save_path, "input_enc_len")) == False:
        os.mkdir(os.path.join(args.pre_data_save_path, "input_enc_len"))
    if os.path.exists(os.path.join(args.pre_data_save_path, "input_decoder")) == False:
        os.mkdir(os.path.join(args.pre_data_save_path, "input_decoder"))

    with tqdm(total=3000) as pbar:
        for i, (src, indices) in enumerate(loader):
            pbar.update(1)
            src, src_length = src
            if src_length[0] > args.max_seq_len:
                f_test_en.readline()
                f_test_de.readline()
                continue
            else:
                src_np = np.append(src.numpy().reshape(-1), np.zeros(args.max_seq_len-src_length[0])).astype(np.int32)
                src_np.tofile(os.path.join(args.pre_data_save_path, "input_encoder/" + "in_" + str(count) + '.bin'))
                src_length_np = src_length[0].numpy().astype(np.int32)
                src_length_np.tofile(os.path.join(args.pre_data_save_path, "input_enc_len/" + "in_" + str(count) + '.bin'))
                bos_np.tofile(os.path.join(args.pre_data_save_path, "input_decoder/" + "in_" + str(count) +'.bin'))

                f_test_en_len.write(f_test_en.readline())
                f_test_de_len.write(f_test_de.readline())
                count += 1
    print("{} bin files generated successfully.".format(count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./gnmt.pth')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--pre_data_save_path', default='./pre_data/')
    parser.add_argument('--max_seq_len', type=int, default=30)
    args = parser.parse_args()

    run_preprocess(args)


if __name__ == '__main__':
    main()
