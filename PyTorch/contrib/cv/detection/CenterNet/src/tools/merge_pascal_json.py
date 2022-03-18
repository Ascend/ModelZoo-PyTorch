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

import json

# ANNOT_PATH = '/home/zxy/Datasets/VOC/annotations/'
ANNOT_PATH = 'voc/annotations/'
OUT_PATH = ANNOT_PATH
INPUT_FILES = ['pascal_train2012.json', 'pascal_val2012.json',
               'pascal_train2007.json', 'pascal_val2007.json']
OUTPUT_FILE = 'pascal_trainval0712.json'
KEYS = ['images', 'type', 'annotations', 'categories']
MERGE_KEYS = ['images', 'annotations']

out = {}
tot_anns = 0
for i, file_name in enumerate(INPUT_FILES):
  data = json.load(open(ANNOT_PATH + file_name, 'r'))
  print('keys', data.keys())
  if i == 0:
    for key in KEYS:
      out[key] = data[key]
      print(file_name, key, len(data[key]))
  else:
    out['images'] += data['images']
    for j in range(len(data['annotations'])):
      data['annotations'][j]['id'] += tot_anns
    out['annotations'] += data['annotations']
    print(file_name, 'images', len(data['images']))
    print(file_name, 'annotations', len(data['annotations']))
  tot_anns = len(out['annotations'])
print('tot', len(out['annotations']))
json.dump(out, open(OUT_PATH + OUTPUT_FILE, 'w'))
