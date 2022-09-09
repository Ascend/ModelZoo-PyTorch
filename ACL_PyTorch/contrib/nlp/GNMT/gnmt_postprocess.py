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
import glob
import argparse
import json
import numpy as np
import re
import torch
import subprocess
from tqdm import tqdm
sys.path.append('./DeepLearningExamples/PyTorch/Translation/GNMT/')
from seq2seq.data.dataset import RawTextDataset
from seq2seq.inference.translator import gather_predictions
from seq2seq.data.tokenizer import Tokenizer

def run_postprocess(args):
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    tokenizer = Tokenizer()
    tokenizer.set_state(checkpoint['tokenizer'])


    if os.path.exists(args.res_file_path) == False:
        os.mkdir(args.res_file_path)
    f_pred = open(os.path.join(args.res_file_path, "pred_sentences.txt"), "wt")
    
    bin_file_num = len(glob.glob(os.path.join(args.bin_file_path, "*.bin")))

    with open(os.path.join(args.bin_file_path, "sumary.json"), "r") as f:
        sumary = json.load(f)

    out_data = dict()
    for item in sumary["filesinfo"].values():
        input_no = re.search(r"in_(\d+)\.bin", item["infiles"][0]).groups()[-1]
        out_data[input_no] = item["outfiles"][0]
    
    with tqdm(total=len(out_data)) as pbar:
        for i in range(len(out_data)):
            pbar.update(1)
            preds = np.fromfile(out_data[str(i)], dtype=np.int32).reshape([1, -1])

            is_end = np.where(preds == 3)
            if is_end[0].size:
                for i in range(is_end[0].size):
                    preds[is_end[0][i]][is_end[1][i]+1:] = 0

            output = []
            for pred in preds:
                pred = pred.tolist()
                detok = tokenizer.detokenize(pred)
                output.append(detok)
            lines = [line + '\n' for line in output]
            f_pred.writelines(lines)
    f_pred.close()
    print("finished!")
    print("The translation is stored in: " + os.path.join(args.res_file_path, "pred_sentences.txt"))


def run_score(args):
    result_file = os.path.join(args.res_file_path, "pred_sentences.txt")
    expected_file = os.path.join(args.pre_file_path, "test_de.txt")
    sacrebleu_params = '--score-only -lc --tokenize intl'
    sacrebleu = subprocess.run([f'sacrebleu --input {result_file} \
                                {expected_file} {sacrebleu_params}'],
                                stdout=subprocess.PIPE, shell=True)
    test_bleu = round(float(sacrebleu.stdout.strip()), 2)
    print("BLEU score:", test_bleu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./gnmt.pth')
    parser.add_argument('--bin_file_path', default='./out_data/')
    parser.add_argument('--res_file_path', default='./res_data/')
    parser.add_argument('--pre_file_path', default='./pre_data/')
    args = parser.parse_args()

    run_postprocess(args)
    run_score(args)


if __name__ == '__main__':
    main()
