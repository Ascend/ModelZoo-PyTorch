# Copyright 2020 Huawei Technologies Co., Ltd
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
from feeder.feeder_kinetics import Feeder_kinetics
import os
import sys
import pickle
import argparse

import numpy as np
from numpy.lib.format import open_memmap

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

toolbar_width = 30


def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(
        data_path,
        label_path,
        data_out_path,
        label_out_path,
        num_person_in=5,  # observe the first 5 persons
        num_person_out=2,  # then choose 2 persons with the highest score
        max_frame=300):

    feeder = Feeder_kinetics(data_path=data_path,
                             label_path=label_path,
                             num_person_in=num_person_in,
                             num_person_out=num_person_out,
                             window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp = open_memmap(data_out_path,
                     dtype='float32',
                     mode='w+',
                     shape=(len(sample_name), 3, max_frame, 18,
                            num_person_out))

    for i, s in enumerate(sample_name):
        data, label = feeder[i]
        print_toolbar(
            i * 1.0 / len(sample_name),
            '({:>5}/{:<5}) Processing data: '.format(i + 1, len(sample_name)))
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument('--data_path',
                        default='data/Kinetics/kinetics-skeleton')
    parser.add_argument('--out_folder',
                        default='data/Kinetics/kinetics-skeleton')
    arg = parser.parse_args()

    part = ['train', 'val']
    for p in part:
        data_path = '{}/kinetics_{}'.format(arg.data_path, p)
        label_path = '{}/kinetics_{}_label.json'.format(arg.data_path, p)
        data_out_path = '{}/{}_data.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        gendata(data_path, label_path, data_out_path, label_out_path)
