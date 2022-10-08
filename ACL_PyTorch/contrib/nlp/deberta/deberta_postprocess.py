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
# ============================================================================

import csv
import os
import numpy as np
import argparse
import time
from collections import OrderedDict, defaultdict
from sklearn.metrics import accuracy_score, f1_score


#============================================================================
# Functions
#============================================================================
def get_labels():
    """See base class."""
    return ["contradiction", "neutral", "entailment"]

def label2id(labelstr):
    label_dict = {l: i for i, l in enumerate(get_labels())}
    return label_dict[labelstr] if labelstr in label_dict else -1

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def metrics_fn(logits, labels, genres):
    metrics = OrderedDict(accuracy=metric_accuracy(logits, labels))
    genres_predicts = defaultdict(list)
    for g, lg, lab in zip(genres, logits, labels):
        genres_predicts[g].append((lg, lab))
    for k in genres_predicts:
        logits_ = [x[0] for x in genres_predicts[k]]
        labels_ = [x[1] for x in genres_predicts[k]]
        acc = metric_accuracy(logits_, labels_)
        metrics[f'accuracy_{k}'] = acc
        metrics['eval_samples'] = len(labels)
    return metrics

def metric_accuracy(logits, labels):
    predicts = np.argmax(logits, axis=1)
    return accuracy_score(labels, predicts)

def run_postprocess(args):
    input_version = os.path.join(args.datasets_path, "dev_" + args.dataset_version + "ed.tsv")
    assert os.path.exists(input_version), f"{input_version} doesn't exists"
    data = _read_tsv(input_version)
    genres = [l[3] for l in data[1:]]
    labels = list(label2id(l[-1]) for l in data[1:])
    labels = np.asarray(labels, dtype=np.int32)

    result=OrderedDict()
    predicts = []
    pre_files=os.listdir(args.bin_file_path)
    pre_files.sort(key=lambda x:int(x[6:-13]))

    for pre_file in pre_files:
        pre = np.fromfile(os.path.join(args.bin_file_path,pre_file),dtype=np.float32).reshape(-1,3)
        predicts.append(pre)

    predicts = np.asarray(predicts, dtype=np.float32).reshape(-1,3)
    predicts = predicts[0:labels.shape[0], :]

    metrics = metrics_fn(predicts, labels, genres)
    result.update(metrics)
    
    with open(os.path.join(args.eval_save_path, args.eval_save_file), 'w', encoding='utf-8') as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))


#============================================================================
# Main
#============================================================================
if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path', default='./MNLI/')
    parser.add_argument('--bin_file_path', default='./result/outputs_bs1_om/')
    parser.add_argument('--dataset_version', choices=['match', 'mismatch'])
    parser.add_argument('--eval_save_path', default='./result/')
    parser.add_argument('--eval_save_file', default='./result_bs1_match.txt')
    args = parser.parse_args()

    run_postprocess(args)

    elapsed = (time.time() - start)
    print("Time used:", elapsed, "s")
