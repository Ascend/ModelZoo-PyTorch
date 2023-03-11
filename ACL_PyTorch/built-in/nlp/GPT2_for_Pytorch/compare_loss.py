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
import json
import random
import transformers
import torch
import numpy as np
import argparse
from datetime import datetime
from ais_bench.infer.interface import InferSession
from torch.nn import CrossEntropyLoss


random.seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type=int, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_path', default='./om/gpt2_4bs_sim_modify.om', type=str, required=False, help='模型的路径')
    parser.add_argument('--tokenized_data_path', default='data/tokenized_eval/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='batch size')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次')
    parser.add_argument('--n_ctx', default=512, type=int, required=False, help='文字长度')
    parser.add_argument('--stride', default=768, type=int, required=False, help='取数据的窗口步长')
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--output_dir', default='eval_result_npu/', type=str, required=False, help='结果输出路径')

    args = parser.parse_args()
    device_id = args.device
    session = InferSession(device_id = device_id, model_path = args.model_path)
    tokenized_data_path = args.tokenized_data_path
    batch_size = args.batch_size
    log_step = args.log_step
    stride = args.stride
    num_pieces = args.num_pieces
    output_dir = args.output_dir
    n_ctx = args.n_ctx

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    overall_step = 0
    total_loss = 0
    total_steps = 0
    # eval
    now = datetime.now()
    print('[INFO] time: {}'.format(now))
    piece_num = 0
    for i in range(num_pieces):
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            line = f.read().strip()
        tokens = line.split()
        tokens = [int(token) for token in tokens]
        start_point = 0
        samples = []
        while start_point < len(tokens) - n_ctx:
            samples.append(tokens[start_point: start_point + n_ctx])
            start_point += stride
        start_point -= stride
        random.shuffle(samples)
        for step in range(len(samples) // batch_size):  # drop last

            #  prepare data
            batch = samples[step * batch_size: (step + 1) * batch_size]
            batch_labels = []
            batch_inputs = []
            for ids in batch:
                int_ids_for_labels = [int(x) for x in ids]
                int_ids_for_inputs = [int(x) for x in ids]
                batch_labels.append(int_ids_for_labels)
                batch_inputs.append(int_ids_for_inputs)
            batch_labels = np.array(batch_labels).astype(np.int64)
            batch_inputs = np.array(batch_labels).astype(np.int64)
            #  forward pass
            feeds = [ batch_inputs ]
            outputs = session.infer(feeds)
            pred = outputs[0]
            #  get loss
            lm_logits = torch.from_numpy(pred)
            shift_logits = lm_logits[..., :-1, :].contiguous().float()
            labels = torch.from_numpy(batch_labels)
            shift_labels = labels[..., 1:].contiguous()
           
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss
            total_steps += 1

            if (overall_step + 1) % log_step == 0:
                print('[INFO] now time: {}:{}. Step {} of piece {}, ppl {}'.format(
                    datetime.now().hour,
                    datetime.now().minute,
                    (step + 1),
                    piece_num,
                    torch.exp(loss)))
        piece_num += 1
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        with open(args.output_dir + 'result.txt', 'w') as f:
            f.write(
                f'{np.exp(total_loss.detach().numpy() / total_steps)}'
                )
    
    print(f"NPU的FPS: \
        {1000 * batch_size / np.mean(session.sumary().exec_time_list)}"
        )

if __name__ == '__main__':
    main()
