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
"""TinyBERT finetuning runner specifically for SST-2 dataset."""

################## import libraries ##################

#standard libraries
from __future__ import absolute_import, division, print_function
import argparse
import random

#third-party libraries
import numpy as np
import torch

#local libraries
from transformer.modeling import TinyBertForSequenceClassification

################## end import libraries ##################

def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

def is_same(a,b,i):
    result = (a == b).mean()
    if result == 1:
        print("step {} = step {}: {}".format(i-1,i,'True'))
    else:
        print("step {} = step {}: {}".format(i - 1, i, 'False'))

def main():

    ################## set args ##################
    parser = argparse.ArgumentParser()

    # 1.file and model
    parser.add_argument("--model",
                        default=None,
                        type=str,
                        help="The model dir.")
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    args = parser.parse_args()

    # create model
    model = TinyBertForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.eval()
    # test
    input_ids = torch.tensor(random_int_list(0,9999,args.max_seq_length), dtype=torch.long).view(1,-1)
    print(input_ids)
    segment_ids = torch.tensor(random_int_list(0,1,args.max_seq_length), dtype=torch.long).view(1,-1)
    input_mask = torch.tensor(random_int_list(0,1,args.max_seq_length), dtype=torch.long).view(1,-1)
    repeat_time = 20
    for i in range(1,repeat_time+1):
        logits, _, _ = model(input_ids, segment_ids, input_mask)
        logits = logits.squeeze()
        print("step {}, logits = {}".format(i,logits))
        if i == 1:
            a = logits
        elif i == 2:
            b = logits
            is_same(a.detach().numpy(),b.detach().numpy(),i)
        else:
            a = b
            b = logits
            is_same(a.detach().numpy(),b.detach().numpy(),i)

if __name__ == "__main__":
    main()
