# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
import sys
import numpy as np
from tqdm import tqdm
from datasets import load_metric


def compute_metrics(preds, labels, eval_metric_path="./accuracy.py"):
    metric = load_metric(eval_metric_path)
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


def load_data(input_path, shape, dtype="int64"):
    return np.fromfile(input_path, dtype=dtype).reshape(shape)


def postprocess(result_dir_out, gt_dir_out, bs, seq_len, num_data):
    all_logits = []
    all_labels = []
    for step in tqdm(range(num_data)):
        data_path = os.path.join(result_dir_out, "{}_0.bin".format(step))
        label_path = os.path.join(gt_dir_out, "{}.bin".format(step))
        logits = load_data(data_path, [bs, seq_len])
        labels = load_data(label_path, [bs, seq_len])
        all_logits.append(logits)
        all_labels.append(labels)
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metric = compute_metrics(all_logits, all_labels)
    print(metric)



if __name__ == '__main__':
    result_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    seq_length = int(sys.argv[3])
    batch_size = 1
    data_files = list(filter(lambda x: os.path.splitext(x)[1]==".bin", os.listdir(result_dir)))
    num_datas = len(data_files)
    num_labels = len(os.listdir(gt_dir))
    assert num_datas == num_labels

    postprocess(result_dir, gt_dir, batch_size, seq_length, num_datas)
