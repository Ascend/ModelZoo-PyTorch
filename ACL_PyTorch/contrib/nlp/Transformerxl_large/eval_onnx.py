# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
import argparse
import time
import math
import os, sys

import torch
import onnx
import onnxruntime
from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import get_logger

class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))
 
    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_input_feed(self, input_name, data, target, mems):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        
        if mems == None or mems == []:
            
            for i, name in enumerate(input_name):
                if i == 0:
                    input_feed[name] = data.numpy()
                else:
                    input_feed[name] = target.numpy()
        else:
            print('mems in not None')
        
        return input_feed
 
    def forward(self, data, target, mems):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, data, target, mems)
        ret = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return ret
 

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/enwik8',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='enwik8',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=128,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=128,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, default='./LM-TFM-enwik8/20211216-100033',
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda" if args.cuda else "cpu")

# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

# Load dataset
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)

va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)

logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
       args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))


###############################################################################
# Evaluation code
###############################################################################
def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_len, total_loss = 0, 0.
    start_time = time.time()
    with torch.no_grad():
        mems = tuple()
        for idx, (data, target, seq_len) in enumerate(eval_iter):
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.item()
            total_len += seq_len
            if idx <= 10:
                print('batch {} ==========>loss_model: {}'.format(idx,loss))
            else:
                break
        total_time = time.time() - start_time
    logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx+1)))
    return total_loss / total_len
    
def evaluate_onnx(eval_iter):    
    onnx_path = 'model.onnx'
    onnx_model = ONNXModel(onnx_path)
    total_len, total_loss = 0, 0.
    start_time = time.time()
    
    #mems = list()    
    #for i in range(25):
        #mems.append(torch.zeros(args.mem_len, args.batch_size, 1024).to(device))
    mems = None
    with torch.no_grad():
        for idx, (data, target, seq_len) in enumerate(eval_iter):
            ts = time.perf_counter()
            ret = onnx_model.forward(data,target, mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.item()
            total_len += seq_len
            # if idx <= 10:
            print('batch {} use time {:.2f}ms ==========>loss_onnx: {:.4f}'.format(idx,(time.perf_counter()-ts)*1000,loss))
            # else:
            #     break
        total_time = time.time() - start_time
    logging('Time : {:.2f}s, {:.2f}ms/segment'.format(
            total_time, 1000 * total_time / (idx+1)))
    return total_loss / total_len
    

# Run on test data.
if args.split == 'all':
    test_loss = evaluate(te_iter)
    valid_loss = evaluate(va_iter)
elif args.split == 'valid':
    valid_loss = evaluate_onnx(va_iter)
    test_loss = None
elif args.split == 'test':
    test_loss = evaluate_onnx(te_iter)
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
