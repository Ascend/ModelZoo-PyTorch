"""
Copyright 2023 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import glob
import os
import json
import numpy as np
from tqdm import tqdm

import torch

from jasper import config
from common import helpers
from common.helpers import process_evaluation_epoch
from jasper.model import GreedyCTCDecoder


def get_parser():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Relative model config path given dataset floder')
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Save predictions in text form at this location')
    parser.add_argument('--save_bin', type=str, default=None,
                        help='Output bin files at this location')
    parser.add_argument('--json_file', type=str, default="./agg_txts.json",
                        help='json file name')
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    cfg = config.load(args.model_config)
    symbols = helpers.add_ctc_blank(cfg['labels'])

    with open(args.json_file, "r") as f:
        agg = json.load(f)

    greedy_decoder = GreedyCTCDecoder()

    files = glob.glob(os.path.join(args.save_bin, "*_0.bin"))
    files.sort()

    for f in tqdm(files):
        log_probs = np.fromfile(f, dtype=np.float32).reshape(1,2000,29)

        preds = greedy_decoder(torch.tensor(log_probs))

        agg['preds'] += helpers.gather_predictions([preds], symbols)
        agg['logits'].append(log_probs)

    wer, _ = process_evaluation_epoch(agg)
    print(f'eval_wer: {100 * wer}')

    if args.save_predictions:
        with open(args.save_predictions, 'w') as f:
            f.write('\n'.join(agg['preds']))


if __name__ == "__main__":
    main()
