
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
# ============================================================================import os
import scipy.io as io
from tqdm import tqdm

gt_mat_path = 'data/SynthText/gt.mat'
im_root = 'data/SynthText/'
txt_root = 'data/SynthText/gt/'

if not os.path.exists(txt_root):
    os.mkdir(txt_root)

print('reading data from {}'.format(gt_mat_path))
gt = io.loadmat(gt_mat_path)
print('Done.')

for i, imname in enumerate(tqdm(gt['imnames'][0])):
    imname = imname[0]
    img_id = os.path.basename(imname)
    im_path = os.path.join(im_root, imname)
    txt_path = os.path.join(txt_root, img_id.replace('jpg', 'txt'))

    if len(gt['wordBB'][0,i].shape) == 2:
        annots = gt['wordBB'][0,i].transpose(1, 0).reshape(-1, 8)
    else:
        annots = gt['wordBB'][0,i].transpose(2, 1, 0).reshape(-1, 8)
    with open(txt_path, 'w') as f:
        f.write(imname + '\n')
        for annot in annots:
            str_write = ','.join(annot.astype(str).tolist())
            f.write(str_write + '\n')

print('End.')