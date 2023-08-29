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

import os
import json
import time
import argparse

import open_clip
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def clip_score(model_clip, tokenizer, preprocess, prompt, image_files, device):
    imgs = []
    texts = []
    for image_file in image_files:
        img = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
        imgs.append(img)
        text = tokenizer([prompt]).to(device)
        texts.append(text)

    img = torch.cat(imgs)   # [bs, 3, 224, 224]
    text = torch.cat(texts) # [bs, 77]

    with torch.no_grad():
        text_ft = model_clip.encode_text(text).float()
        img_ft = model_clip.encode_image(img).float()
        score = F.cosine_similarity(img_ft, text_ft).squeeze()
    
    return score.cpu()


def main():
    args = parse_arguments()
    
    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)
    
    t_b = time.time()
    print(f"Load clip model...") 
    model_clip, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.model_weights_path, device=device)
    model_clip.eval()
    print(f">done. elapsed time: {(time.time() - t_b):.3f} s")
    
    tokenizer = open_clip.get_tokenizer(args.model_name)

    with os.fdopen(os.open(args.image_info, os.O_RDONLY), "r") as f:
        image_info = json.load(f)

    t_b = time.time()
    print(f"Calc clip score...") 
    all_scores = []
    cat_scores = {}

    for i, info in enumerate(image_info):
        image_files = info['images']
        category = info['category']
        prompt = info['prompt']

        print(f"[{i + 1}/{len(image_info)}] {prompt}")

        image_scores = clip_score(model_clip, 
                                  tokenizer, 
                                  preprocess, 
                                  prompt, 
                                  image_files, 
                                  device)
        if len(image_files) > 1:
            best_score = max(image_scores)
        else:
            best_score = image_scores

        print(f"image scores: {image_scores}")
        print(f"best score: {best_score}")

        all_scores.append(best_score)
        if category not in cat_scores:
            cat_scores[category] = []
        cat_scores[category].append(best_score)
    print(f">done. elapsed time: {(time.time() - t_b):.3f} s")

    average_score = np.average(all_scores)
    print(f"====================================")
    print(f"average score: {average_score:.3f}")
    print(f"category average scores:")
    cat_average_scores = {}
    for category, scores in cat_scores.items():
        cat_average_scores[category] = np.average(scores)
        print(f"[{category}], average score: {cat_average_scores[category]:.3f}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="device for torch.",
    )
    parser.add_argument(
        "--image_info",
        type=str,
        default="./image_info.json",
        help="Image_info.json file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-H-14",
        help="open clip model name",
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
        help="open clip model weights",
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()