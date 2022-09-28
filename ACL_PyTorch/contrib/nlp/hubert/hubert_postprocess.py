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
sys.path.append('./fairseq/')
import glob
import argparse
import numpy as np
import torch
import logging
import json
import editdistance
import fairseq
from typing import List, Dict, Any, Tuple
from fairseq.data.data_utils import post_process
from fairseq.data.dictionary import Dictionary
from fairseq import utils
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode(
    emissions: torch.FloatTensor,
) -> List[List[Dict[str, torch.LongTensor]]]:
    def get_pred(e):
        toks = e.argmax(dim=-1).unique_consecutive()
        return toks[toks != 0]

    return [[{"tokens": get_pred(x), "score": 0}] for x in emissions]

def process_sentence(
    sample: Dict[str, Any],
    hypo: Dict[str, Any],
    tgt_dict : Dictionary,
) -> Tuple[int, int]:
    toks = sample

    # Processes hypothesis.
    hyp_pieces = tgt_dict.string(hypo["tokens"].int().cpu())
    hyp_words = post_process(hyp_pieces, "letter")

    # Processes target.
    target_tokens = utils.strip_pad(toks, tgt_dict.pad())
    tgt_pieces = tgt_dict.string(target_tokens.int().cpu())
    tgt_words = post_process(tgt_pieces, "letter")

    logger.info(f"HYPO: {hyp_words}")
    logger.info(f"REF: {tgt_words}")
    logger.info("---------------------")

    hyp_words, tgt_words = hyp_words.split(), tgt_words.split()

    return editdistance.eval(hyp_words, tgt_words), len(tgt_words)


def read_info_from_json(json_path):
    if os.path.exists(json_path) is False:
        print(json_path, 'is not exist')
    with open(json_path, 'r') as f:
        load_data = json.load(f)
        file_info = load_data['filesinfo']
        return file_info


def run_postprocess(args):
    file_info = read_info_from_json(args.source_json_path)

    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.model_path])

    if os.path.exists(args.res_file_path) == False:
        os.makedirs(args.res_file_path)

    total_errors = 0
    total_length = 0
    for i in file_info.items():
        source_path = i[1]['outfiles'][0]
        label_id = os.path.basename(i[1]['infiles'][0])[6:-4]
        source = torch.Tensor(np.fromfile(os.path.join(source_path), dtype=np.float32)).reshape(1812,1,32)
        label = torch.Tensor(np.fromfile(os.path.join(args.label_bin_file_path, "label" + str(label_id) + ".bin"), dtype=np.int32))
        source = source.transpose(0, 1).float().cpu().contiguous()
        hypos = decode(source)

        errs, length = process_sentence(sample=label,hypo=hypos[0][0],tgt_dict = task.target_dictionary)

        total_errors += errs
        total_length += length
        print("total_errors",total_errors,"total_length",total_length)
    print("avg:",total_errors*1.0/total_length*100)
    output = "total_errors:" + str(total_errors) + " total_length:" + str(total_length) + " AVG:" + str(total_errors * 1.0 / total_length * 100)
    f_pred = open(os.path.join(args.res_file_path, "error_rate.txt"), "wt")
    f_pred.writelines(output)
    f_pred.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./data/pt/hubert_large_ll60k_finetune_ls960.pt')
    parser.add_argument('--source_json_path', default='.out_data/*/sumary.json')
    parser.add_argument('--label_bin_file_path', default='./pre_data/test-clean/label/')
    parser.add_argument('--res_file_path', default='./res_data/test-clean/')

    args = parser.parse_args()

    run_postprocess(args)


if __name__ == '__main__':
    main()