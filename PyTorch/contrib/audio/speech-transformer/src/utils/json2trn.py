#!/usr/bin/env python
# encoding: utf-8
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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging
from utils import process_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str, help='json files')
    parser.add_argument('dict', type=str, help='dict')
    parser.add_argument('ref', type=str, help='ref')
    parser.add_argument('hyp', type=str, help='hyp')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.info("reading %s", args.json)
    with open(args.json, 'r') as f:
        j = json.load(f)

    logging.info("reading %s", args.dict)
    char_list, sos_id, eos_id = process_dict(args.dict)
    # with open(args.dict, 'r') as f:
    #     dictionary = f.readlines()
    # char_list = [unicode(entry.split(' ')[0], 'utf_8') for entry in dictionary]
    # char_list.insert(0, '<blank>')
    # char_list.append('<eos>')
    # print([x.encode('utf-8') for x in char_list])

    logging.info("writing hyp trn to %s", args.hyp)
    logging.info("writing ref trn to %s", args.ref)
    h = open(args.hyp, 'w')
    r = open(args.ref, 'w')

    for x in j['utts']:
        seq = [char_list[int(i)] for i in j['utts'][x]
               ['output'][0]['rec_tokenid'].split()]
        h.write(" ".join(seq).replace('<eos>', '')),
        h.write(
            " (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

        seq = [char_list[int(i)] for i in j['utts'][x]
               ['output'][0]['tokenid'].split()]
        r.write(" ".join(seq).replace('<eos>', '')),
        r.write(
            " (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")
