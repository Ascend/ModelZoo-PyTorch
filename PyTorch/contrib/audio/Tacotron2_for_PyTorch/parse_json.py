#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, getopt
import json
import csv


def load_file():
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:",["ifile="])
    except getopt.GetoptError:
       print('parse_json.py -i <inputfile>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('parse_json.py -i <inputfile>')
          sys.exit()
       elif opt in ("-i", "--ifile"):
          inputfile = arg
    return inputfile


def parse_prof(prof_str):
    prof_dict = json.loads(prof_str)
    dur_list = []
    dur_dict = {}
    count_dict = {}
    with open('prof.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=["name","ph","ts","dur","tid","pid","args","id","cat"])
        w.writeheader()
        for row in prof_dict:
            if not 'dur' in row:
                continue
            w.writerow(row)

            if not row['name'] in dur_dict:
                dur_dict[row['name']] = row['dur']
                count_dict[row['name']] = 1
            else:
                dur_dict[row['name']] += row['dur']
                count_dict[row['name']] += 1
        for op in dur_dict:
            comb_dict = {}
            comb_dict['name'] = op
            comb_dict['dur'] = dur_dict[op]
            comb_dict['count'] = count_dict[op]
            if 'dur' in comb_dict:
                dur_list.append(comb_dict)
    with open('statistic.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, dur_list[0].keys())
        w.writeheader()
        for row in dur_list:
            w.writerow(row)


if __name__ == "__main__":
    inputfile = load_file()
    with open(inputfile) as f:
        line = f.readline()
        parse_prof(line)
        while line:
            line = f.readline()
