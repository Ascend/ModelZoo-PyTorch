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

from __future__ import absolute_import

import os
import sys
import argparse
sys.path.append(os.getcwd())

from got10k.experiments import *

from siamfc import SiamFCTracker

model_path = './models/siamfc_50.pth'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=" SiamFC Test")
    parser.add_argument('--model_path', default=model_path, type=str, help=" the path of model")
    args = parser.parse_args()

    tracker = SiamFCTracker(args.model_path)  # init a track

    root_dir = os.path.abspath('./data/OTB')
    e = ExperimentOTB(root_dir, version=2015)

    e.run(tracker, visualize=False)  # without visualize

    prec_score, succ_score, succ_rate = e.report([tracker.name])
    
    ss = '-prec_score: %.3f -succ_score: %.3f -succ_rate: %.3f' % \
         (float(prec_score), float(succ_score), float(succ_rate))

    print(args.model_path + " : " + ss)
