# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.

 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. """
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import inception_utils
import utils


def prepare_parser():
    usage = 'Calculate and store inception metrics.'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--dataset', type=str, default='I128_hdf5',
        help='Which Dataset to train on, out of I128, I256, C10, C100...'
             'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
    parser.add_argument(
        '--data_root', type=str, default='data',
        help='Default location where data is stored (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Default overall batchsize (default: %(default)s)')
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Train with multiple GPUs (default: %(default)s)')
    parser.add_argument(
        '--augment', action='store_true', default=False,
        help='Augment with random crops and flips (default: %(default)s)')
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of dataloader workers (default: %(default)s)')
    parser.add_argument(
        '--shuffle', action='store_true', default=False,
        help='Shuffle the data? (default: %(default)s)')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use.')
    return parser


def run(config):
    # Get loader
    config['drop_last'] = False
    loaders = utils.get_data_loaders(**config)

    # Load inception net
    device = 'npu:0'
    net = inception_utils.load_inception_net(parallel=config['parallel'], device=device)
    pool, logits, labels = [], [], []

    for i, (x, y) in enumerate(tqdm(loaders[0])):
        x = x.to(device)
        with torch.no_grad():
            pool_val, logits_val = net(x)
            pool += [np.asarray(pool_val.cpu())]
            logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
            labels += [np.asarray(y.cpu())]

    pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
    # uncomment to save pool, logits, and labels to disk
    # print('Saving pool, logits, and labels to disk...')
    # np.savez(config['dataset']+'_inception_activations.npz',
    #           {'pool': pool, 'logits': logits, 'labels': labels})
    # Calculate inception metrics and report them
    print('Calculating inception metrics...')
    IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
    print('Training data from dataset %s has IS of %5.5f +/- %5.5f' % (config['dataset'], IS_mean, IS_std))
    # Prepare mu and sigma, save to disk. Remove "hdf5" by default
    # (the FID code also knows to strip "hdf5")
    print('Calculating means and covariances...')
    mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
    print('Saving calculated means and covariances to disk...')
    np.savez(config['dataset'].strip('_hdf5') + '_inception_moments.npz', **{'mu': mu, 'sigma': sigma})


def main():
    # parse command line
    parser = prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)


if __name__ == '__main__':
    main()
