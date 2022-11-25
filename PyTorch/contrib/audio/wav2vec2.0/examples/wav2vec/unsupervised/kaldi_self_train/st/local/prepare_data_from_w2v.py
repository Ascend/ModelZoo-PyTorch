# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

import kaldi_io
import numpy as np
import os


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("w2v_dir", help="wav2vec feature and text directory")
    parser.add_argument("tar_root", help="output data directory in kaldi's format")
    parser.add_argument("split", help="name of the subset")
    parser.add_argument("--label", default="", help="if specified, copy labels too")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    tar_dir = os.path.join(args.tar_root, args.split)
    os.makedirs(tar_dir, exist_ok=True)

    lengths_path = os.path.join(args.w2v_dir, f"{args.split}.lengths")
    with open(lengths_path) as f:
        lengths = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengths[:-1]).tolist()
    feats = np.load(
        os.path.join(args.w2v_dir, f"{args.split}.npy"),
        mmap_mode="r"
    )
    assert feats.shape[0] == sum(lengths), \
        f"lengths mismatch {feats.shape[0]} != {sum(lengths)}"

    ark_path = os.path.join(tar_dir, "feats.ark")
    scp_path = os.path.join(tar_dir, "feats.scp")
    wspec = f"ark:| copy-feats --compress=true ark:- ark,scp:{ark_path},{scp_path}"
    with kaldi_io.open_or_fd(wspec, "wb") as f:
        for idx, (offset, length) in enumerate(zip(offsets, lengths)):
            feat = feats[offset:offset+length]
            kaldi_io.write_mat(f, feat, key=f"utt{idx:010d}")

    u2s_path = os.path.join(tar_dir, "utt2spk")
    s2u_path = os.path.join(tar_dir, "spk2utt")
    with open(u2s_path, "w") as f_u2s, open(s2u_path, "w") as f_s2u:
        for idx in range(len(lengths)):
            f_u2s.write(f"utt{idx:010d} utt{idx:010d}\n")
            f_s2u.write(f"utt{idx:010d} utt{idx:010d}\n")

    if bool(args.label):
        lab_path = os.path.join(args.w2v_dir, f"{args.split}.{args.label}")
        txt_path = os.path.join(tar_dir, "text")
        with open(lab_path) as f_lab, open(txt_path, "w") as f_txt:
            for idx, line in enumerate(f_lab):
                f_txt.write(f"utt{idx:010d} {line}")

if __name__ == "__main__":
    main()
