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
#


import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from allennlp.nn import util
from allennlp.common.plugins import import_plugins
from allennlp.models.archival import load_archive


def build_archive(archive_file):
    # Load archive
    return load_archive(
        archive_file,
        weights_file=None,
        cuda_device=-1,
        overrides=''
    )


def evaluate(archive, result_dir, label_dir, label_weight_dir):
    model = archive.model
    model.eval()

    for data_idx in tqdm(range(len(os.listdir(label_dir)))):
        label_path = os.path.join(label_dir, f"{data_idx}.npy")
        label_weight_path = os.path.join(label_weight_dir, f"{data_idx}.npy")
        logits_path = os.path.join(result_dir, f"{data_idx}_0.npy")
        if not os.path.exists(logits_path):
            logits_path = os.path.join(result_dir, f"{data_idx}_0.bin")
            logits = torch.tensor(np.fromfile(logits_path, dtype="float32").reshape([-1, 10026]))
        else:
            logits = torch.tensor(np.load(logits_path))
        label = torch.tensor(np.load(label_path))
        label_weights = torch.tensor(np.load(label_weight_path))

        if label is not None and label_weights is not None:
            label_mask = label > 1
            weighted_labels = util.masked_index_replace(
                logits.new_zeros(logits.size() + (1,)),
                label.clamp(min=0),
                label_mask,
                label_weights.unsqueeze(-1),
            ).squeeze(-1)

            binary_label_mask = weighted_labels.new_ones(logits.size())
            binary_label_mask[:, 0] = 0
            binary_label_mask[:, 1] = 0

            model.f1_metric(logits, weighted_labels, binary_label_mask.bool())
            model.vqa_metric(logits, label, label_weights)

    metrics = model.get_metrics()
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Postprocess for vilbert')
    parser.add_argument('--archive_file', help='Path for archive file.')
    parser.add_argument('--result_dir', help='Infer result dir.')
    parser.add_argument('--label_dir', help='GT label dir.')
    parser.add_argument('--label_weight_dir', help='GT label weight dir.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    import_plugins()
    archive = build_archive(args.archive_file)
    metrics = evaluate(
        archive,
        args.result_dir,
        args.label_dir,
        args.label_weight_dir
    )
    print(metrics)
