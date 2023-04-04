# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import torch
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from bert4torch.layers import CRF
import torch.nn as nn
from bert4torch.models import build_transformer_model, BaseModel


class Model(BaseModel):
    def __init__(self, args):
        super().__init__()
        self.bert = build_transformer_model(config_path=args.config_path, checkpoint_path=None, segment_vocab_size=0)
        # embedding_dims:768, len_categories: 7
        self.fc = nn.Linear(768, 7)  # 包含首尾
        self.crf = CRF(7)


def load_bin_file(path, shape, dtype="float32"):
    data = np.fromfile(path, dtype).reshape(shape)
    return torch.tensor(data)


def evaluate(result_dir, label_dir):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    X2, Y2, Z2 = 1e-10, 1e-10, 1e-10
    true_labels, true_predictions = [], []
    data_num = len(os.listdir(label_dir))
    for data_idx in tqdm(range(data_num)):
        emission_score_path = os.path.join(
            result_dir, "{}_0.bin".format(data_idx))
        attention_mask_path = os.path.join(
            result_dir, "{}_1.bin".format(data_idx))
        label_path = os.path.join(
            label_dir, "{}.bin".format(data_idx))
        labels = load_bin_file(label_path, [1, 256], dtype="int64")
        emission_score = torch.Tensor(
            load_bin_file(emission_score_path, [1, 256, 7]))
        attention_mask = torch.Tensor(
            load_bin_file(attention_mask_path, [1, 256], dtype="int64"))
        scores = crf.decode(emission_score, attention_mask)
        true_labels += [[categories_id2label[int(l)] for
                         l in label if l != -100] for label in labels]
        true_predictions += [[categories_id2label[int(p)] for
                              p in score if p != -100] for score in scores]

        attention_mask = labels.gt(0)
        # token粒度
        X += (scores.eq(labels) * attention_mask).sum().item()
        Y += scores.gt(0).sum().item()
        Z += labels.gt(0).sum().item()

        # entity粒度
        entity_pred = trans_entity2tuple(scores)
        entity_true = trans_entity2tuple(labels)
        X2 += len(entity_pred.intersection(entity_true))
        Y2 += len(entity_pred)
        Z2 += len(entity_true)
    eval_result = classification_report(true_labels,
                                        true_predictions,
                                        digits=4,
                                        mode='strict',
                                        scheme=IOB2)
    print(eval_result)
    f1, p1, r1 = 2 * X / (Y + Z), X / Y, X / Z
    f2, p2, r2 = 2 * X2 / (Y2 + Z2), X2 / Y2, X2 / Z2
    print("val-token level: f1:{}, precision: {}, recall:{}".format(f1, p1, r1))
    print("val-entity level: f1:{}, precision: {}, recall:{}".format(f2, p2, r2))
    return eval_result, f1, p1, r1, f2, p2, r2


def trans_entity2tuple(scores):
    '''把tensor转为(样本id, start, end, 实体类型)的tuple用于计算指标
    '''
    batch_entity_ids = set()
    for i, one_samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(one_samp):
            flag_tag = categories_id2label[item.item()]
            if flag_tag.startswith('B-'):  # B
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif len(entity_ids) == 0:
                continue
            elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('I-') and \
                 (flag_tag[2:] == entity_ids[-1][-1]):  # I
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                entity_ids.append([])

        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))
    return batch_entity_ids


def parse_arguments():
    parser = argparse.ArgumentParser(description='SwinTransformer onnx export.')
    parser.add_argument('-i', '--result_dir', type=str, required=True,
                        help='result dir for prediction results')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='save path for evaluation result')
    parser.add_argument('-l', '--label_dir', type=str, required=True,
                        help='label dir for label results')
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help='config path for export model')
    parser.add_argument('-k', '--ckpt_path', type=str, default="./best_model.pt",
                        help='result dir for prediction results')
    args = parser.parse_args()
    args.out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
    categories_id2label = {i: k for i, k in enumerate(categories)}
    model = Model(args).to("cpu")
    model.load_weights(args.ckpt_path, strict=False)
    crf = model.crf

    seqeval_result, f1_score, precision, recall, \
        f2_score, precision2, recall2 = evaluate(args.result_dir, args.label_dir)
    evaluate_results = {
        "seqeval_result": seqeval_result,
        "val-token  level": {
            "f1": f1_score,
            "precision": precision,
            "recall": recall
        },
        "val-entity level": {
            "f1": f2_score,
            "precision": precision2,
            "recall": recall2
        }
    }
    with open(args.out_path, 'w') as f:
        json.dump(evaluate_results, f, ensure_ascii=False, indent=4)
