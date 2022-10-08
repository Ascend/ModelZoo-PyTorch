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
import sys
sys.path.append('./fairseq')
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import fairseq
from fairseq.data.audio import hubert_dataset
from fairseq.data import data_utils
from fairseq.tasks.hubert_pretraining import LabelEncoder
import itertools
from tqdm import tqdm

def postprocess(wav):
    if wav.dim() == 2:
        wav = wav.mean(-1)
    assert wav.dim() == 1, wav.dim()
    wav = F.layer_norm(wav, wav.shape)
    return wav


def get_audio(audio_root, audio_names, index):
    import soundfile as sf

    wav_path = os.path.join(audio_root, audio_names[index])
    wav, cur_sample_rate = sf.read(wav_path)
    wav = torch.from_numpy(wav).float()
    wav = postprocess(wav)
    return wav


def get_label(label_processors, label_paths,label_offsets_list, index, label_idx):
    with open(label_paths[label_idx]) as f:
        offset_s, offset_e = label_offsets_list[label_idx][index]
        f.seek(offset_s)
        label = f.read(offset_e - offset_s)

    if label_processors is not None:
        label = label_processors[label_idx](label)
    return label


def getitem(audio_root,audio_names, label_processors, label_paths, label_offsets_list, index):
    wav = get_audio(audio_root, audio_names, index)
    labels = get_labels(label_processors, label_paths, label_offsets_list, index)
    return {"id": index, "source": wav, "label_list": labels}


def get_labels(label_processors, label_paths, label_offsets_list, index):
    return [get_label(label_processors, label_paths, label_offsets_list, index, i) for i in range(1)]


def ordered_indices(sizes):
    order = [np.random.permutation(len(sizes))]
    order.append(sizes)
    return np.lexsort(order)[::-1]


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def run_preprocess(args):
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.model_path])

    if os.path.exists(args.pre_data_source_save_path) == False:
        os.makedirs(args.pre_data_source_save_path)
    if os.path.exists(args.pre_data_label_save_path) == False:
        os.makedirs(args.pre_data_label_save_path)

    label_paths = []
    label_paths.append(args.datasets_ltr_path)
    audio_root, audio_names, inds, tot, sizes = hubert_dataset.load_audio(args.datasets_tsv_path, None, None)
    label_processors = [LabelEncoder(task.target_dictionary)]
    label_offsets_list = [load_label_offset(p, inds, tot) for p in label_paths]

    with data_utils.numpy_seed(1):
        indices = ordered_indices(sizes)

    for i in tqdm(range (len(indices))):
        sample = getitem(audio_root, audio_names, label_processors, label_paths, label_offsets_list, indices[i])
        sample["source"] = sample["source"].view(1, -1)
        len_source = len(sample["source"][0].cpu())
        add_source = torch.zeros(1, 580000 - len_source).float().to("cpu")
        sample["source"] = torch.cat((sample["source"], add_source), 1)

        np_source = np.array(sample["source"])
        np_label = np.array(sample["label_list"][0])

        np_source.tofile(os.path.join(args.pre_data_source_save_path + "source" + str(i) + '.bin'))
        np_label.tofile(os.path.join(args.pre_data_label_save_path + "label" + str(i) + '.bin'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./data/pt/hubert_large_ll60k_finetune_ls960.pt')
    parser.add_argument('--datasets_tsv_path', default='./data/test-clean/train.tsv')
    parser.add_argument('--datasets_ltr_path', default='./data/test-clean/train.ltr')
    parser.add_argument('--pre_data_source_save_path', default='./pre_data/test-clean/source/')
    parser.add_argument('--pre_data_label_save_path', default='./pre_data/test-clean/label/')
    args = parser.parse_args()

    run_preprocess(args)


if __name__ == '__main__':
    main()