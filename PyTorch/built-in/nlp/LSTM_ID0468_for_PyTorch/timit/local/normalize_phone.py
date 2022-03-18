#encoding=utf-8
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

import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Normalize the phoneme on TIMIT")
parser.add_argument("--map", default="./decode_map_48-39/phones.60-48-39.map", help="The map file")
parser.add_argument("--to", default=48, help="Determine how many phonemes to map")
parser.add_argument("--src", default='./data_prepare/train/phn_text', help="The source file to mapping")
parser.add_argument("--tgt", default='./data_prepare/train/48_text' ,help="The target file after mapping")

def main():
    args = parser.parse_args()
    if not os.path.exists(args.map) or not os.path.exists(args.src):
        print("Map file or source file not exist !")
        sys.exit(1)
    
    map_dict = {}
    with open(args.map) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if args.to == "60-48":
                if len(line) == 1:
                    map_dict[line[0]] = ""
                else:
                    map_dict[line[0]] = line[1]
            elif args.to == "60-39": 
                if len(line) == 1:
                    map_dict[line[0]] = ""
                else:
                    map_dict[line[0]] = line[2]
            elif args.to == "48-39":
                if len(line) == 3:
                    map_dict[line[1]] = line[2]
            else:
                print("%s phonemes are not supported" % args.to)
                sys.exit(1)
    
    with open(args.src, 'r') as rf, open(args.tgt, 'w') as wf:
        for line in rf.readlines():
            line = line.strip().split(' ')
            uttid, utt = line[0], line[1:]
            map_utt = [ map_dict[phone] for phone in utt if map_dict[phone] != "" ]
            wf.writelines(uttid + ' ' + ' '.join(map_utt) + '\n')

if __name__ == "__main__":
    main()
