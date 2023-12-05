# Copyright 2023 Huawei Technologies Co., Ltd
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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='dataset path',
                        default='data/coco')
    args = parser.parse_args()
    with open(args.dataset + "annotations/instances_minival2014.json", 'r') as file:
        content = file.read()
    content = json.loads(content)
    info = content.get('info')
    licenses = content.get('licenses')
    images = content.get('images')
    print(len(images))
    annotations = content.get('annotations')
    categroies = content.get('categories')

    with open('./coco2014.names', 'w') as f:
        for categroie in categroies:
            f.write(categroie.get('name').replace(' ', '_'))
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

    assert len(annotation_ids) == len(bboxs) == len(category_ids) ==len(segmentations) # 255094

    with open('coco_2014.info', 'w') as f:
        for index, file_name in enumerate(file_names):
            file_name = args.dataset + "images/val2014/" + file_name
            line = "{} {} {} {}".format(index, file_name, widths[index], heights[index])
            f.write(line)
            f.write('\n')

    def get_all_index(lst, item):
        return [index for (index, value) in enumerate(lst) if value == item]

    def get_categroie_name(lst, item):
        categroie_name =  [dt.get('name') for dt in lst if item == dt.get('id')][0]
        if len(categroie_name.split()) == 2:
            temp = categroie_name.split()
            categroie_name = temp[0] + '_' + temp[1]
        return categroie_name

    for index, image_id in enumerate(image_ids):
        indexs = get_all_index(annotation_ids, image_id)
        with open('./ground-truth/{}.txt'.format(file_names[index].split('.')[0]), 'w') as f:
            for idx in indexs:
                f.write(get_categroie_name(categroies, category_ids[idx]))
                
                f.write(' ')
                # change label
                bboxs[idx][2] = bboxs[idx][0] + bboxs[idx][2]
                bboxs[idx][3] = bboxs[idx][1] + bboxs[idx][3]
                f.write(' '.join(map(str, bboxs[idx])))
                f.write('\n')




