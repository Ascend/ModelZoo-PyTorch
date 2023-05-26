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
import argparse
import os 

from pycocotools.coco import COCO 
import numpy
import torchvision.transforms as transforms
from PIL import Image
import tqdm

def preprocess(image):
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=normalization_mean,
                                     std=normalization_std)
    trans_val = transforms.Compose([
                transforms.Resize((300,300)),
                transforms.ToTensor(),
                normalize,])
    return trans_val(image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess of SSD pytorch model')
    parser.add_argument("--val_annotation", default="bbox_only_instances_val2017.json")
    parser.add_argument("--data_root", default="./coco/val2017")
    parser.add_argument("--save_path", default="ssd_bin")
    args = parser.parse_args()
    val_annotation = args.val_annotation
    data_root = args.data_root
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    cocoGT = COCO(annotation_file=val_annotation)
    data = cocoGT.dataset
    images = {}
    label_map = {}
    cnt = 0
    for cat in data['categories']:
        cnt += 1
        label_map[cat['id']] = cnt
        
    for img in data["images"]:
        img_id = img["id"]
        img_name = img["file_name"]
        img_size = (img["height"], img["width"])
        images[img_id] = (img_name, img_size, [])

    for bboxes in data["annotations"]:
        img_id = bboxes["image_id"]
        category_id = bboxes["category_id"]
        bbox = bboxes["bbox"]
        bbox_label = label_map[bboxes["category_id"]]
        images[img_id][2].append((bbox, bbox_label))


    for k, v in tqdm.tqdm(list(images.items())):
        if len(v[2]) == 0:
            images.pop(k)
        else:
            file_path = os.path.join(data_root, v[0])
            img = Image.open(file_path).convert("RGB")
            img = preprocess(img)
            img = img.numpy()
            img.tofile(os.path.join(save_path, v[0].split('.')[0] + '.bin'))