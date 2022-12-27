# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================

from pathlib import Path
from time import strftime
import torch

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

INPUT_KEYS = ['input_ids', 'attention_mask', 'token_type_ids']
NAME_ENTITY = {k: v for v, k in enumerate([
    'O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'
])}


def generate(model_name_from_hub='./bert-large-NER', input_file='./conll2003/test.txt', save_dir=None):
    if save_dir is None:
        save_dir = f'./bert_bin/bert_bin_{strftime("%Y%m%d-%H%M%S")}'
        save_dir_input_ids = save_dir + '/input_ids'
        save_dir_attention_mask = save_dir + '/attention_mask'
        save_dir_token_type_ids = save_dir + '/token_type_ids'
        
                       
    input_file = Path(input_file)
    if not input_file.is_file():
        print('Input files not exist')
        return
    
    save_dir = Path(save_dir)
    save_dir_input_ids = Path(save_dir_input_ids)
    save_dir_attention_mask = Path(save_dir_attention_mask)
    save_dir_token_type_ids = Path(save_dir_token_type_ids)
    if save_dir.is_dir():
        print('Save directory is already exist')
        return
    else:
        save_dir.mkdir(parents=True)
        save_dir_input_ids.mkdir(parents=True)
        save_dir_attention_mask.mkdir(parents=True)
        save_dir_token_type_ids.mkdir(parents=True)
    
    anno_file = save_dir / '..' / (save_dir.name + '.anno')

    inputs = {k: [] for k in INPUT_KEYS}
    tokenizer = AutoTokenizer.from_pretrained(model_name_from_hub)
    with input_file.open() as infile, anno_file.open(mode='w') as annofile:
        idx = 0
        sentence_words = []
        sentence_tags = []
        input_ids = []
        token_type_ids = []
        attention_mask = []
        
        for line in tqdm(infile):
            line = line.strip('\n')
            if line == '-DOCSTART- -X- -X- O':
                continue
            if line == '':     
                if len(sentence_words) : 
                    for i in range(len(sentence_words)):
                        if sentence_words[i].isupper():
                            sentence_words[i]=sentence_words[i].lower()

                    tokens = tokenizer.convert_tokens_to_ids(sentence_words)

                    input_ids.append(np.pad(tokens, (0, 512-len(tokens)), 'constant'))
                    token_type_ids.append([0]*512)
                    attention_mask.append([1]*len(tokens)+[0]*(512-len(tokens)))
                    
                    np.array(input_ids).tofile(save_dir_input_ids / f'input_ids_{idx}_{idx}.bin')
                    np.array(token_type_ids).tofile(save_dir_token_type_ids / f'token_type_ids_{idx}_{idx}.bin')
                    np.array(attention_mask).tofile(save_dir_attention_mask / f'attention_mask_{idx}_{idx}.bin')

                    annofile.write(f'{idx}')
                    
                    for tag in sentence_tags:
                        annofile.write(f' ')
                        annofile.write(f'{NAME_ENTITY[tag]}')
                        
                    annofile.write('\n')
                    
                    sentence_words.clear()
                    sentence_tags.clear()
                    input_ids.clear()
                    token_type_ids.clear()
                    attention_mask.clear()
               
                    idx += 1
                    
            else:
                
                word, _, _, tag = line.split(' ')
                sentence_words.append(word)
                sentence_tags.append(tag)


if __name__ == '__main__':
    generate()