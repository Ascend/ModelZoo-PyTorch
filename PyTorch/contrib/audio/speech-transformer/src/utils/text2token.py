#!/usr/bin/env python2
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

import sys
import argparse
import re


def exist_or_not(i, match_pos):
    start_pos = None
    end_pos = None
    for pos in match_pos:
        if pos[0] <= i < pos[1]:
            start_pos = pos[0]
            end_pos = pos[1]
            break

    return start_pos, end_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nchar', '-n', default=1, type=int,
                        help='number of characters to split, i.e., \
                        aabb -> a a b b with -n 1 and aa bb with -n 2')
    parser.add_argument('--skip-ncols', '-s', default=0, type=int,
                        help='skip first n columns')
    parser.add_argument('--space', default='<space>', type=str,
                        help='space symbol')
    parser.add_argument('--non-lang-syms', '-l', default=None, type=str,
                        help='list of non-linguistic symobles, e.g., <NOISE> etc.')
    parser.add_argument('text', type=str, default=False, nargs='?',
                        help='input text')
    args = parser.parse_args()

    rs = []
    if args.non_lang_syms is not None:
        with open(args.non_lang_syms, 'r') as f:
            nls = [unicode(x.rstrip(), 'utf_8') for x in f.readlines()]
            rs = [re.compile(re.escape(x)) for x in nls]

    if args.text:
        f = open(args.text)
    else:
        f = sys.stdin
    line = f.readline()
    n = args.nchar
    while line:
        x = unicode(line, 'utf_8').split()
        print ' '.join(x[:args.skip_ncols]).encode('utf_8'),
        a = ' '.join(x[args.skip_ncols:])

        # get all matched positions
        match_pos = []
        for r in rs:
            i = 0
            while i >= 0:
                m = r.search(a, i)
                if m:
                    match_pos.append([m.start(), m.end()])
                    i = m.end()
                else:
                    break

        if len(match_pos) > 0:
            chars = []
            i = 0
            while i < len(a):
                start_pos, end_pos = exist_or_not(i, match_pos)
                if start_pos is not None:
                    chars.append(a[start_pos:end_pos])
                    i = end_pos
                else:
                    chars.append(a[i])
                    i += 1
            a = chars

        a = [a[i:i + n] for i in range(0, len(a), n)]

        a_flat = []
        for z in a:
            a_flat.append("".join(z))

        a_chars = [z.replace(' ', args.space) for z in a_flat]
        print ' '.join(a_chars).encode('utf_8')
        line = f.readline()


if __name__ == '__main__':
    main()
