# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import argparse
import json
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--output-name', type=str, default='val_subset.json',
                        help='name of output file with subset of val labels')
    parser.add_argument('--num-images', type=int, default=250, help='number of images in subset')
    args = parser.parse_args()

    with open(args.labels, 'r') as f:
        data = json.load(f)

    random.seed(0)
    total_val_images = 5000
    idxs = list(range(total_val_images))
    random.shuffle(idxs)

    images_by_id = {}
    for idx in idxs[:args.num_images]:
        images_by_id[data['images'][idx]['id']] = data['images'][idx]

    annotations_by_image_id = {}
    for annotation in data['annotations']:
        if annotation['image_id'] in images_by_id:
            if not annotation['image_id'] in annotations_by_image_id:
                annotations_by_image_id[annotation['image_id']] = []
            annotations_by_image_id[annotation['image_id']].append(annotation)

    subset = {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': [],
        'annotations': [],
        'categories': data['categories']
    }
    for image_id, image in images_by_id.items():
        subset['images'].append(image)
        if image_id in annotations_by_image_id:  # image has at least 1 annotation
            subset['annotations'].extend(annotations_by_image_id[image_id])

    with open(args.output_name, 'w') as f:
        json.dump(subset, f, indent=4)

