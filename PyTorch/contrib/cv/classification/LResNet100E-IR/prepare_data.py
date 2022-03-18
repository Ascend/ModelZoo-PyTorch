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
from pathlib import Path
from config import get_config
from data.data_pipe import load_bin, load_mx_rec
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-r", "--rec_path", help="mxnet record file path",default='faces_emore', type=str)
    args = parser.parse_args()
    conf = get_config()
    rec_path = conf.data_path/args.rec_path
    load_mx_rec(rec_path)
    
    bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    
    for i in range(len(bin_files)):
        load_bin(rec_path/(bin_files[i]+'.bin'), rec_path/bin_files[i], conf.test_transform)