#
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
#
''' Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the 
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.
 
 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import inception_utils
from tqdm import tqdm, trange
from argparse import ArgumentParser
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')

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
  net = inception_utils.load_inception_net(parallel=config['parallel'])
  pool, logits, labels = [], [], []
  device = 'npu'
  for i, (x, y) in enumerate(tqdm(loaders[0])):
    x = x.to(f'npu:{NPU_CALCULATE_DEVICE}')
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
  np.savez(config['dataset'].strip('_hdf5')+'_inception_moments.npz', **{'mu' : mu, 'sigma' : sigma})

def main():
  # parse command line    
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)


if __name__ == '__main__':    
    main()