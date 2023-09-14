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

import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from clip_benchmark.metrics.zeroshot_retrieval import batchify, recall_at_k


def evaluate(vision_embed_dir, text_embed_dir, recall_k_list=None):
    """reference: https://github.com/LAION-AI/CLIP_benchmark/"""
    vision_embed_list = []
    text_embed_list = []

    # for each text, we collect the corresponding image index, as each images
    # can have multiple corresponding texts
    texts_image_index = []

    for i, npy_file in tqdm(enumerate(os.listdir(vision_embed_dir))):
        vision_embed_path = osp.join(vision_embed_dir, npy_file)
        vision_embed = torch.from_numpy(np.load(vision_embed_path))
        prefix = osp.splitext(npy_file)[0].rsplit('_', 1)[0]
        text_embeds = []
        for text_index in range(5):
            text_embed_path = osp.join(text_embed_dir, f'{prefix}_{text_index}_0.npy')
            text_embeds.append(torch.from_numpy(np.load(text_embed_path)))
        text_embeds = torch.cat(text_embeds, dim=0)

        vision_embed = F.normalize(vision_embed, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        vision_embed_list.append(vision_embed)
        text_embed_list.append(text_embeds)
        texts_image_index.extend([torch.tensor(i, dtype=torch.int64)] * 5)

    # concatenate all embeddings
    all_vision_embed = torch.cat(vision_embed_list)
    all_text_embed = torch.cat(text_embed_list)

    # get the score for each text and image pair
    scores = all_text_embed @ all_vision_embed.t()

    # construct a positive pair matrix, which tells whether each text-image
    # pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True

    if not recall_k_list:
        recall_k_list = [5]
    metrics = {}
    for recall_k in recall_k_list:
        metrics[f'image_retrieval_recall@{recall_k}'] = (batchify(
            recall_at_k, scores, positive_pairs, 1, 'cpu', k=recall_k) > 0
        ).float().mean().item()
        metrics[f'text_retrieval_recall@{recall_k}'] = (batchify(
            recall_at_k, scores.T, positive_pairs.T, 1, 'cpu', k=recall_k) > 0
        ).float().mean().item()

    print(f'\nMetrics: {metrics}')
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Postprocess.')
    parser.add_argument('--vision-embeds', type=str,
                        default='./om_outputs/vision_embed_bs8',
                        help='Path to vision model inference results.')
    parser.add_argument('--text-embeds', type=str,
                        default='./om_outputs/text_embed_bs8',
                        help='Path to text model inference results.')
    args = parser.parse_args()

    evaluate(args.vision_embeds, args.text_embeds)


if __name__ == '__main__':
    main()
