# Copyright 2021 Huawei Technologies Co., Ltd
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
import os
import json
import argparse
from tqdm import tqdm


def get_all_index(lst, item):
    return [index for (index, value) in enumerate(lst) if value == item]


def get_categroie_name(lst, item):
    categroie_name = [dt.get('name') for dt in lst if item == dt.get('id')][0]
    if len(categroie_name.split()) == 2:
        temp = categroie_name.split()
        categroie_name = temp[0] + '_' + temp[1]
    return categroie_name


def main(args):
    with open(args.annotation_path, 'r') as file:
        content = file.read()
    content = json.loads(content)
    info = content.get('info')
    licenses = content.get('licenses')
    images = content.get('images')
    annotations = content.get('annotations')
    categroies = content.get('categories')

    # generate names file
    names_path = args.names_path
    with open(names_path, 'w') as f:
        for categroie in categroies:
            categroie_name = categroie.get('name')
            if len(categroie_name.split()) == 2:
                temp = categroie_name.split()
                categroie_name = temp[0] + '_' + temp[1]
            f.write(categroie_name)
            f.write('\n')

    file_names = [image.get('file_name') for image in images]
    widths = [image.get('width') for image in images]
    heights = [image.get('height') for image in images]
    image_ids = [image.get('id') for image in images]
    assert len(file_names) == len(widths) == len(heights) == len(image_ids), "must be equal"

    annotation_ids = [annotation.get('image_id') for annotation in annotations]
    bboxs = [annotation.get('bbox') for annotation in annotations]
    category_ids = [annotation.get('category_id') for annotation in annotations]
    segmentations = [annotation.get('segmentation') for annotation in annotations]
    iscrowds = [annotation.get('iscrowd') for annotation in annotations]

    assert len(annotation_ids) == len(bboxs) == len(category_ids) == len(segmentations)  # 255094

    # generate info file
    with open(args.coco_info_path, 'w') as f:
        for index, file_name in enumerate(file_names):
            file_name = 'val2017/' + file_name
            line = "{} {} {} {}".format(index, file_name, widths[index], heights[index])
            f.write(line)
            f.write('\n')

    # generate txt split file
    gt_file_path = args.ground_truth_dir
    if not os.path.exists(gt_file_path):
        os.makedirs(gt_file_path)

    for index, image_id in tqdm(enumerate(image_ids)):
        indexs = get_all_index(annotation_ids, image_id)
        with open('{}/{}.txt'.format(gt_file_path, file_names[index].split('.')[0]), 'w') as f:
            for idx in indexs:
                f.write(get_categroie_name(categroies, category_ids[idx]))
                f.write(' ')
                # change label
                bboxs[idx][2] = bboxs[idx][0] + bboxs[idx][2]
                bboxs[idx][3] = bboxs[idx][1] + bboxs[idx][3]
                f.write(' '.join(map(str, bboxs[idx])))
                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YoloV5 annotation parse.')
    parser.add_argument('--annotation_path', type=str, default="./instances_val2017.json", help='annotation file path')
    parser.add_argument('--names_path', type=str, default="./coco_2017.names", help='class name save path')
    parser.add_argument('--coco_info_path', type=str, default="./coco_2017.info", help='coco info path')
    parser.add_argument('--ground_truth_dir', type=str, default="./ground-truth-split",
                        help='split ground truth save path')
    flags = parser.parse_args()

    main(flags)
