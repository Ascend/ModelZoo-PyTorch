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

import sys
import json
import argparse
from pathlib import Path

import tqdm
import numpy as np
import torch
import torch.nn as nn

import config
from crf import CRF
from utils_ner import get_entities
from ner_metrics import SeqEntityScore


class Postprocessor:
    def __init__(self, transitions_npy):
        self.label2id = config.label2id
        self.id2label = {i: label for i, label \
                         in enumerate(self.label2id)}
        self.crf = self.build_crf(transitions_npy)

    def build_crf(self, transitions_npy):
        crf = CRF(tagset_size=len(self.label2id), 
                  tag_dictionary=self.label2id, 
                  device=torch.device("cpu"))
        state_dict = {
            'transitions': torch.from_numpy(np.load(transitions_npy))
        }
        crf.load_state_dict(state_dict)
        crf.eval()
        return crf

    def postprocess(self, features, input_lens):
        num_texts = len(input_lens)
        if features.size(0) > num_texts:
            features = features[:num_texts, ...]

        tags_batch, _ = self.crf._obtain_labels(
                            features, self.id2label, input_lens)
        label_entities = []
        for tags in tags_batch:
            label_entities.append(get_entities(tags, self.id2label))
        return tags_batch, label_entities

    @staticmethod
    def write_results(orig_batch, entities_batch, writer):
        save_lines = []
        for item, entities in zip(orig_batch, entities_batch):
            if 'lable' in item:
                item['anno_label'] = item.pop('label')
            item['pred_label'] = {}
            words = list(item['text'])

            if len(entities) != 0:
                for subject in entities:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    word = "".join(words[start:end + 1])
                    if tag in item['pred_label']:
                        if word in item['pred_label'][tag]:
                            item['pred_label'][tag][word].append([start, end])
                        else:
                            item['pred_label'][tag][word] = [[start, end]]
                    else:
                        item['pred_label'][tag] = {}
                        item['pred_label'][tag][word] = [[start, end]]
            line = json.dumps(item, ensure_ascii=False) + '\n'
            writer.write(line)

    def __call__(self, features, input_lens):
        return self.postprocess(features, input_lens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_results", type=str, 
                        help='path to results of inference.')
    parser.add_argument("--annotations", type=str, help='path to label file.')
    args = parser.parse_args()

    anno_info = {}
    with open(args.annotations, 'r') as f:
        for line in f:
            info = json.loads(line.strip())
            data_index = info.pop('data_index')
            anno_info[data_index] = info
    
    postprocessor = Postprocessor(transitions_npy="inference/transitions.npy")
    metric = SeqEntityScore(postprocessor.id2label, markup='bios')
    for path in tqdm.tqdm(Path(args.infer_results).iterdir()):
        features = torch.from_numpy(np.load(str(path)))
        data_index = int(path.stem.split('_')[1])
        info = anno_info[data_index]
        pred_tags, label_entities = postprocessor(features, info['lens'])
        metric.update(pred_paths=pred_tags, label_paths=info['tags'])

    eval_info, class_info = metric.result()
    print('metrics:')
    print(eval_info)
    print('metrics of per category:')
    print(json.dumps(class_info, ensure_ascii=True, indent=4))


if __name__ == "__main__":
    main()
