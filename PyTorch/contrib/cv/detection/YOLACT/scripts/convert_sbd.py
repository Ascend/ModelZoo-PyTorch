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
import scipy.io, scipy.ndimage
import os.path, json
import pycocotools.mask
import numpy as np

def mask2bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax - cmin, rmax - rmin



inst_path = './inst/'
img_path  = './img/'
img_name_fmt = '%s.jpg'
ann_name_fmt = '%s.mat'

image_id = 1
ann_id   = 1

types = ['train', 'val']

for t in types:
    with open('%s.txt' % t, 'r') as f:
        names = f.read().strip().split('\n')

    images = []
    annotations = []

    for name in names:
        img_name = img_name_fmt % name

        ann_path = os.path.join(inst_path, ann_name_fmt % name)
        ann = scipy.io.loadmat(ann_path)['GTinst'][0][0]

        classes = [int(x[0]) for x in ann[2]]
        seg = ann[0]

        for idx in range(len(classes)):
            mask = (seg == (idx + 1)).astype(np.float)

            rle = pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('ascii')

            annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': classes[idx],
                'segmentation': rle,
                'area': float(mask.sum()),
                'bbox': [int(x) for x in mask2bbox(mask)],
                'iscrowd': 0
            })

            ann_id += 1
        
        img_name = img_name_fmt % name
        img = scipy.ndimage.imread(os.path.join(img_path, img_name))

        images.append({
            'id': image_id,
            'width': img.shape[1],
            'height': img.shape[0],
            'file_name': img_name
        })

        image_id += 1

    info = {
        'year': 2012,
        'version': 1,
        'description': 'Pascal SBD',
    }

    categories = [{'id': x+1} for x in range(20)]

    with open('pascal_sbd_%s.json' % t, 'w') as f:
        json.dump({
            'info': info,
            'images': images,
            'annotations': annotations,
            'licenses': {},
            'categories': categories
        }, f)

