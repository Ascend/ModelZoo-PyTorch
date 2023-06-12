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

# coding:utf-8


import os
import json
import re
import numpy as np
import argparse

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
from torchvision.datasets.utils import download_url


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


def preprocess(src_path, img_path, ids_path, mask_path):

    ann_root = 'annotation'
    if not os.path.exists(ann_root):
        os.mkdir(ann_root)
    
    filenames = {'val': 'coco_karpathy_val.json',
                 'test': 'coco_karpathy_test.json'}
    annotation = json.load(
        open(os.path.join(ann_root, filenames['test']), 'r'))

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        normalize, ])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    txt_id = 0
    for img_id, ann in enumerate(annotation):
        image = ann['image']
        input_image = Image.open(os.path.join(src_path, image)).convert('RGB')
        input_tensor = transform_test(input_image)
        img_np = np.array(input_tensor).astype(np.float32)
        img_np.tofile(os.path.join(img_path, str(img_id) + ".bin"))

        for i, caption in enumerate(ann['caption']):
            txt = pre_caption(caption, 30)
            txt = tokenizer(txt, padding='max_length',
                            truncation=True, max_length=35, return_tensors="pt")
            input_ids = txt.input_ids
            input_mask = txt.attention_mask
            input_ids_np = input_ids.numpy()
            input_mask_np = input_mask.numpy()
            input_ids_np.tofile(os.path.join(
                ids_path, str(txt_id) + '.bin'))
            input_mask_np.tofile(os.path.join(
                mask_path, str(txt_id) + '.bin'))
            txt_id += 1

    print("generate bin runs successfuly!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', default='/opt/npu/dcc/coco2014/')
    parser.add_argument('--save_bin_path',
                        default='/opt/npu/dcc/coco2014_bin/')
    args = parser.parse_args()

    coco_path = args.coco_path
    save_bin_path = args.save_bin_path

    save_img_path = os.path.join(save_bin_path, 'img/')
    save_ids_path = os.path.join(save_bin_path, 'ids/')
    save_mask_path = os.path.join(save_bin_path, 'mask/')

    for path in [save_ids_path, save_img_path, save_mask_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    preprocess(coco_path, save_img_path, save_ids_path, save_mask_path)
