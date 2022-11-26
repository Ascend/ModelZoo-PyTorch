"""
Copyright 2021 Huawei Technologies Co., Ltd

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
from numpy.testing._private.utils import measure
from tqdm import tqdm

import torch

from jasper import config
from common import helpers
from common.dataset import (AudioDataset, get_data_loader)
from common.features import FilterbankFeatures
from common.helpers import process_evaluation_epoch
from jasper.model import GreedyCTCDecoder

from acl_net import AclModel


def get_parser():
    parser = argparse.ArgumentParser(description='Jasper')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Data batch size')
    parser.add_argument('--dataset_dir', type=str,
                        help='Absolute path to dataset folder')
    parser.add_argument('--val_manifests', type=str, nargs='+',
                        help='Relative path to evaluation dataset manifest files')
    parser.add_argument('--model', default=None, type=str,
                        help='Path to om model')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Relative model config path given dataset floder')
    parser.add_argument("--max_duration", default=None, type=float,
                        help='maximum duration of sequences. if None uses attribute from model configuration file')
    parser.add_argument("--pad_to_max_duration", action='store_true',
                        help='pad to maximum duration of sequences')
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Save predictions in text form at this location')
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
                                  args.batch_size,
                                  multi_gpu=False,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=False)

    feat_proc = FilterbankFeatures(**features_kw)

    measurements = {}
    om_model = AclModel(device_id=args.device_id,
                        model_path=args.model,
                        sync_infer=args.sync_infer,
                        measurements=measurements,
                        key='per_infer_time_ns',
                        cpu_run=args.cpu_run)

    feat_proc.to('cpu')
    feat_proc.eval()

    agg = {'txts': [], 'preds': [], 'logits': []}

    greedy_decoder = GreedyCTCDecoder()

    for it, batch in enumerate(tqdm(data_loader)):

        batch = [t.to(torch.device('cpu'), non_blocking=True) for t in batch]
        audio, audio_lens, txt, txt_lens = batch
        feats, feat_lens = feat_proc(audio, audio_lens)
        feats = feats.numpy()
        feat_lens = feat_lens.numpy()

        # om infer
        batch_rank = 4000
        inputs = [feats, feat_lens]
        dims = [args.batch_size, 64, batch_rank, args.batch_size]
        dims_info = {'dimCount': 4, 'name': '', 'dims': dims}
        res = om_model(inputs, dims_info)
        # because om has random order, so use if branch to get target result
        for item in res:
            if len(item.shape) == 3:
                log_probs = item
                break

        preds = greedy_decoder(torch.tensor(log_probs))

        if txt is not None:
            agg['txts'] += helpers.gather_transcripts([txt], [txt_lens],
                                                      symbols)
        agg['preds'] += helpers.gather_predictions([preds], symbols)
        agg['logits'].append(log_probs)

    wer, _ = process_evaluation_epoch(agg)
    print(f'eval_wer: {100 * wer}')

    if args.save_predictions:
        with open(args.save_predictions, 'w') as f:
            f.write('\n'.join(agg['preds']))


if __name__ == "__main__":
    main()
