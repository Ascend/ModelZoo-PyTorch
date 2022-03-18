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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

MODEL_PATH = '../../models/ExtremeNet_500000.pkl'
OUT_PATH = '../../models/ExtremeNet_500000.pth'

import torch
state_dict = torch.load(MODEL_PATH)
key_map = {'t_heats': 'hm_t', 'l_heats': 'hm_l', 'b_heats': 'hm_b', \
           'r_heats': 'hm_r', 'ct_heats': 'hm_c', \
           't_regrs': 'reg_t', 'l_regrs': 'reg_l', \
           'b_regrs': 'reg_b', 'r_regrs': 'reg_r'}

out = {}
for k in state_dict.keys():
  changed = False
  for m in key_map.keys():
    if m in k:
      if 'ct_heats' in k and m == 't_heats':
        continue
      new_k = k.replace(m, key_map[m])
      out[new_k] = state_dict[k]
      changed = True
      print('replace {} to {}'.format(k, new_k))
  if not changed:
    out[k] = state_dict[k]
data = {'epoch': 0,
        'state_dict': out}
torch.save(data, OUT_PATH)
