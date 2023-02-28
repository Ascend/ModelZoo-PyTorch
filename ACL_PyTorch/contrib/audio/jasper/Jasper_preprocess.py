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
import os
import json
import numpy as np
from tqdm import tqdm

import torch

from jasper import config
from common import helpers
from common.dataset import (AudioDataset, get_data_loader)
from common.features import FilterbankFeatures


def get_parser():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument('--dataset_dir', type=str,
                        help='Absolute path to dataset folder')
    parser.add_argument('--val_manifests', type=str, nargs='+',
                        help='Relative path to evaluation dataset manifest files')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Relative model config path given dataset floder')
    parser.add_argument("--max_duration", default=None, type=float,
                        help='maximum duration of sequences. if None uses attribute from model configuration file')
    parser.add_argument("--pad_to_max_duration", action='store_true',
                        help='pad to maximum duration of sequences')
    parser.add_argument('--save_bin_0', type=str, default='./prep_data_0',
                        help='Save input 0 in bin format at this location')
    parser.add_argument('--save_bin_1', type=str, default='./prep_data_1',
                        help='Save input 1 in bin format at this location')
    parser.add_argument('--json_file', type=str, default="./agg_txts.json",
                        help='json file name')
    parser.add_argument("--device_id", default=0,
                        type=int, help='Select device for inference')
    parser.add_argument('--cpu_run', default=True, choices=['True', 'False'],
                        help='Whether to run on cpu')
    parser.add_argument('--sync_infer', default=True, choices=['True', 'False'],
                        help='sync or async')
    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()
    path_0 = args.save_bin_0
    path_1 = args.save_bin_1
    if not os.path.exists(path_0):
        os.makedirs(path_0)
    if not os.path.exists(path_1):
        os.makedirs(path_1)

    cfg = config.load(args.model_config)
    if args.max_duration is not None:
        cfg['input_val']['audio_dataset']['max_duration'] = args.max_duration
        cfg['input_val']['filterbank_features']['max_duration'] = args.max_duration

    if args.pad_to_max_duration:
        assert cfg['input_val']['audio_dataset']['max_duration'] > 0
        cfg['input_val']['audio_dataset']['pad_to_max_duration'] = True
        cfg['input_val']['filterbank_features']['pad_to_max_duration'] = True

    symbols = helpers.add_ctc_blank(cfg['labels'])

    dataset_kw, features_kw = config.input(cfg, 'val')

    # dataset
    dataset = AudioDataset(args.dataset_dir,
                           args.val_manifests,
                           symbols,
                           **dataset_kw)

    data_loader = get_data_loader(dataset,
                                  1,
                                  multi_gpu=False,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False)

    feat_proc = FilterbankFeatures(**features_kw)
    feat_proc.to('cpu')
    feat_proc.eval()

    agg = {'txts': [], 'preds': [], 'logits': []}

    for it, batch in enumerate(tqdm(data_loader)):

        batch = [t.to(torch.device('cpu'), non_blocking=True) for t in batch]
        audio, audio_lens, txt, txt_lens = batch
        feats, feat_lens = feat_proc(audio, audio_lens)
        feats = feats.numpy()
        feat_lens = feat_lens.numpy()

        inputs = [feats, feat_lens]
        if txt is not None:
            agg['txts'] += helpers.gather_transcripts([txt], [txt_lens],
                                                      symbols)
            feats_name = "{:0>12d}.bin".format(it)
            lens_name = "{:0>12d}.bin".format(it)
            feats.tofile(os.path.join(path_0, feats_name))
            feat_lens.tofile(os.path.join(path_1, lens_name))
    with open(args.json_file, "w") as f:
        json.dump(agg, f)

if __name__ == "__main__":
    main()
