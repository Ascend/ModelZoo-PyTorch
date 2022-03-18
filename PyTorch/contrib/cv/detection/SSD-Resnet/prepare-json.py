#!/usr/bin/env python

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from argparse import ArgumentParser

parser = ArgumentParser(description="parse coco annotations file and compress out data we don't need for bbox training")

parser.add_argument('input_file', type=str, help='input instances_*.json file')
parser.add_argument('output_file', type=str, help='"compressed" output instances_*.json file')
parser.add_argument('--keep-keys', '-k', action='store_true',
                    help='pycocotools depends on all the useless keys. This flag lets you keep the useless keys (but deletes the data they point to)')
parser.add_argument('--pretty-print', '-p', action='store_true',
                    help='pretty print the output .json file (for debugging) rather than minimizing size')

args = parser.parse_args()

with open(args.input_file) as json_file:
    data = json.load(json_file)

images = data['images']
annots = data['annotations']

print("images: ", len(images))
print("bboxes: ", len(annots))


useful_keys = ['id', 'file_name', 'height', 'width']
for img in images:
    useless_keys = [k for k in img.keys() if k not in useful_keys]
    for uk in useless_keys:
        del img[uk]
#     imgid = i['id']
#     fname = i['file_name']
#     h = i['height']
#     w = i['width']
#     print(imgid, fname, h, w, h*w)

useful_keys = ['image_id', 'bbox', 'category_id', 'id']
for a in annots:
    useless_keys = [k for k in a.keys() if k not in useful_keys]
    if args.keep_keys:
        a['segmentation'] = []  # empty the list
        a['area'] = 1.0         # lie about the area (make it take less space in the file)
    else:
        for uk in useless_keys:
            del a[uk]

newresult = dict()
newresult['images'] = images
newresult['annotations'] = annots
newresult['categories'] = data['categories']
with open(args.output_file, "w+") as ofile:
    my_separators = None if args.pretty_print else (',',':')
    my_indent = 4 if args.pretty_print else None
    json.dump(newresult, ofile, separators=my_separators, indent=my_indent)
#     imid = a['image_id']
#     bbox = a['bbox']
#     cat = a['category_id']
#     myid = a['id']
#     print(imid, myid, bbox[0], bbox[1], bbox[2], bbox[3], cat)

