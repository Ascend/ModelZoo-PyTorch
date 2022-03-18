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
import json
from argparse import ArgumentParser
from utils.dataset import get_train_val_test_split
import os
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))


def main(args):

    train_list, val_list, test_list, label_map = get_train_val_test_split(args.data_root, args.val_list_file, args.test_list_file)

    with open(os.path.join(args.out_dir, "training_list.txt"), "w+") as f:
        f.write("\n".join(train_list))

    with open(os.path.join(args.out_dir, "validation_list.txt"), "w+") as f:
        f.write("\n".join(val_list))

    with open(os.path.join(args.out_dir, "testing_list.txt"), "w+") as f:
        f.write("\n".join(test_list))

    with open(os.path.join(args.out_dir, "label_map.json"), "w+") as f:
        json.dump(label_map, f)

    print("Saved data lists and label map.")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", "--val_list_file", type=str, required=True, help="Path to validation_list.txt.")
    parser.add_argument("-t", "--test_list_file", type=str, required=True, help="Path to test_list.txt.")
    parser.add_argument("-d", "--data_root", type=str, required=True, help="Root directory of speech commands v2 dataset.")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory for data lists and label map.")
    args = parser.parse_args()

    main(args)