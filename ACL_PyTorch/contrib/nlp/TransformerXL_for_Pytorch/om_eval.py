# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#
# coding: utf-8
import argparse
import time
import math
import os, sys
import numpy as np
from tqdm import tqdm
import aclruntime

import torch
import torch.nn.functional as F

from data_utils import get_lm_corpus
from utils.exp_utils import get_logger
from net_infer import Net


parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=5,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, required=True,
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
parser.add_argument('--om_path', type=str, required=True,
                    help='path to the om file')
args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda" if args.cuda else "cpu")

# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

# Load dataset
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)

te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)

logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
       args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

###############################################################################
# Evaluation code
###############################################################################
def evaluate(eval_iter):
    # init infersession
    options = aclruntime.session_options()
    # options.acl_json_path = "./acl.json"

    device_id = 0
    session = aclruntime.InferenceSession(args.om_path, device_id, options)
    # get output name list
    outputs_name = [meta.name for meta in session.get_outputs()]

    # padding for arg0
    inputs = [None]
    # init arg1-arg13 inputs
    for i in range(13):
        ndata = np.zeros((args.mem_len, args.batch_size, 512), dtype=np.float16)
        tensor = aclruntime.Tensor(ndata)
        tensor.to_device(device_id)
        inputs.append(tensor)

    total_len, total_loss = 0, 0.
    start_time = time.time()
    with torch.no_grad():
        for idx, (data, target, seq_len) in tqdm(enumerate(eval_iter)):
            if data.numpy().shape[0] != args.tgt_len:
                print("not enough to process, the number is:", data.numpy().shape[0])
                break
            
            # update arg0 from data
            tensor = aclruntime.Tensor(data.numpy())
            tensor.to_device(device_id)
            inputs[0] = tensor
            
            # inference
            outputs = session.run(outputs_name, inputs)
            # convert outpus0 to host memory and convert numpy
            outputs[0].to_host()
            array0 = np.array(outputs[0])

            # important replace inputs for inference outputs
            for i in range(13):
                inputs[i+1] = outputs[i+1]

            logit = torch.from_numpy(array0.astype(np.float32))
            logit = logit[:target.size(0)]
            loss = -F.log_softmax(logit.view(-1, logit.size(-1)), dim=-1).gather(1, target.view(-1).unsqueeze(1)).squeeze(1)
            loss = loss.mean()
            total_loss += seq_len * loss.item()
            total_len += seq_len
            if idx % 10 == 0:
                print("loss = {:.2f}".format(total_loss / total_len))
        total_time = time.time() - start_time
    logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx+1)))
    s = session.sumary()
    print("throughputRate: {}".format(1000/np.mean(s.exec_time_list)))
    return total_loss / total_len

# Run on test data.
if args.split == 'all':
    test_loss = evaluate(te_iter)
    valid_loss = evaluate(va_iter)
elif args.split == 'valid':
    valid_loss = evaluate(va_iter)
    test_loss = None
elif args.split == 'test':
    test_loss = evaluate(te_iter)
    valid_loss = None

def format_log(loss, split):
    if args.dataset in ['enwik8', 'text8']:
        log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
            split, loss, loss / math.log(2))
    else:
        log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
            split, loss, math.exp(loss))
    return log_str

log_str = ''
if valid_loss is not None:
    log_str += format_log(valid_loss, 'valid')
if test_loss is not None:
    log_str += format_log(test_loss, 'test')

logging('=' * 100)
logging(log_str)
logging('=' * 100)
