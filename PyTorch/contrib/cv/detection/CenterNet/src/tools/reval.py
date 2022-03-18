#!/usr/bin/env python

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


# Reval = re-eval. Re-evaluate saved detections.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), 'voc_eval_lib'))

from model.test import apply_nms
from datasets.pascal_voc import pascal_voc
import pickle
import os, argparse
import numpy as np
import json

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Re-evaluate results')
  parser.add_argument('detection_file', type=str)
  parser.add_argument('--output_dir', help='results directory', type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to re-evaluate',
                      default='voc_2007_test', type=str)
  parser.add_argument('--matlab', dest='matlab_eval',
                      help='use matlab for evaluation',
                      action='store_true')
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                      action='store_true')
  parser.add_argument('--nms', dest='apply_nms', help='apply nms',
                      action='store_true')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def from_dets(imdb_name, detection_file, args):
  imdb = pascal_voc('test', '2007')
  imdb.competition_mode(args.comp_mode)
  imdb.config['matlab_eval'] = args.matlab_eval
  with open(os.path.join(detection_file), 'rb') as f:
    if 'json' in detection_file:
      dets = json.load(f)
    else:
      dets = pickle.load(f, encoding='latin1')
  # import pdb; pdb.set_trace()
  if args.apply_nms:
    print('Applying NMS to all detections')
    test_nms = 0.3
    nms_dets = apply_nms(dets, test_nms)
  else:
    nms_dets = dets

  print('Evaluating detections')
  imdb.evaluate_detections(nms_dets)


if __name__ == '__main__':
  args = parse_args()

  imdb_name = args.imdb_name
  from_dets(imdb_name, args.detection_file, args)
