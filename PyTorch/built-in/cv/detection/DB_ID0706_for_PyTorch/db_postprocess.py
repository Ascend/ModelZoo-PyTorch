# Copyright 2021 Huawei Technologies Co., Ltd
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
import argparse
import cv2
import numpy as np
import torch

from tqdm import tqdm
from experiment import Experiment
from concern.config import Configurable, Config

class Eval:
    def __init__(self, experiment_inp, args_inp, cmd, verbose=False):
        if cmd:
            pass
        else:
            cmd = dict()
        self.experiment = experiment_inp
        experiment_inp.load('evaluation', **args_inp)
        self.data_loaders = experiment_inp.evaluation.data_loaders
        self.args = cmd
        self.logger = experiment_inp.logger
        self.structure = experiment_inp.structure

    @classmethod
    def get_pred(self, filename):
        path_base = os.path.join(flags.bin_data_path, filename.split(".")[0])
        mask_pred = np.fromfile(path_base + "_" + '0' + ".bin", dtype="float32")
        mask_pred = np.reshape(mask_pred, [1, 1, 736, 1280])
        mask_pred = torch.from_numpy(mask_pred)
        return mask_pred

    def eval(self):
        for _, data_loader in self.data_loaders.items():
            raw_metrics = []
            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                pred = self.get_pred(batch['filename'][0])
                output = self.structure.representer.represent(batch, pred)
                raw_metric = self.structure.measurer.validate_measure(batch, output, box_thresh=self.args['box_thresh'])
                raw_metrics.append(raw_metric)
            metrics = self.structure.measurer.gather_measure(raw_metrics, self.logger)
            for key, metric in metrics.items():
                print('%s : %f (%d)' % (key, metric.avg, metric.count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--bin_data_path', default="./result/dumpOutput_device0/")
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    flags = parser.parse_args()

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Eval(experiment, experiment_args, cmd=args).eval()
