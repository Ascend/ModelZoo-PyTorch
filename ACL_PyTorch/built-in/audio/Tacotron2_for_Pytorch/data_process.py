# Copyright 2018 NVIDIA Corporation. All Rights Reserved.
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


import copy
import torch

from tacotron2.text import text_to_sequence

class DataProcess():
    def __init__(self, max_input_len, dynamic, interval) -> None:
        self.max_input_len = max_input_len
        self.dynamic = dynamic
        self.interval = interval
                                        
    def pad_sequences(self, datas, names):
        datas_copy = copy.deepcopy(datas)
        for i in range(len(datas_copy)):
            if len(datas_copy[i]) > self.max_input_len:
                datas[i] = datas[i][:self.max_input_len]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x) for x in datas]),
            dim=0, descending=True)

        if self.dynamic == True:
            input_lengths[0] = (self.interval - input_lengths[0] % self.interval) + input_lengths[0]
        text_padded = torch.LongTensor(len(datas), input_lengths[0])
        text_padded.zero_()
        text_padded[0][:] += torch.IntTensor(text_to_sequence('.', ['english_cleaners'])[:])
        names_new = []
        for i in range(len(ids_sorted_decreasing)):
            text = datas[ids_sorted_decreasing[i]]
            text_padded[i, :text.size(0)] = text
            names_new.append(names[ids_sorted_decreasing[i]])

        return text_padded, input_lengths, names_new 


    def prepare_input_sequence(self, texts, names):
        d = []
        for i,text in enumerate(texts):
            d.append(torch.IntTensor(text_to_sequence(text, ['english_cleaners'])[:]))

        text_padded, input_lengths, names_new = self.pad_sequences(d, names)

        text_padded = text_padded.long()
        input_lengths = input_lengths.long()

        return text_padded, input_lengths, names_new


    def prepare_batch_meta(self, batch_size, meta_data, meta_name):
        batch_data = []
        batch_name = []

        for i in range(batch_size):
            if i == 0:
                batch_data.append(meta_data.pop(0))
                batch_name.append(meta_name.pop(0))
            else:
                batch_data.append(meta_data.pop())
                batch_name.append(meta_name.pop())
        
        if len(batch_data[0]) < self.max_input_len:
            batch_data[0] += ' and'
        return batch_data, batch_name


def get_mask_from_lengths(lengths):
    lengths_tensor = torch.LongTensor(lengths)
    max_len = torch.max(lengths_tensor).item()
    ids = torch.arange(0, max_len, device=lengths_tensor.device, dtype=lengths_tensor.dtype)
    mask = (ids < lengths_tensor.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask


def read_file(input_file):
    meta_data_dict = {}
    if input_file.endswith('.csv'):
        f = open(input_file, encoding='utf-8')
        for line in f:
            meta_data_dict[line.strip().split('|')[0]] = line.strip().split('|')[-1]
    elif input_file.endswith('.txt'):
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                out = line.split('|')
                wav = out[0].split('/', 2)[2].split('.')[0]
                meta_data_dict[wav] = out[1]
    else:
        print("file is not support")

    meta_list = sorted(meta_data_dict.items(), key=lambda value:len(value[1]), reverse=True)
    name_list = [value[0] for value in meta_list]
    value_list = [value[1].strip() for value in meta_list]

    return name_list, value_list




