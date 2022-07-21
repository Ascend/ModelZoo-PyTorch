# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
if torch.__version__ >= "1.8":
    import torch_npu
import apex

from prototype.data.datasets import clip_dataset


# _get_label_text ....-->./DeCLIP
def _get_label_text(self, text):
    # label_text = ['a photo of ' + text + '.']
    if self.label_texts_ensemble == 'prompt6':
        f = f'./DeCLIP/prototype/data/datasets/prompts/query_pattern_prompt6'
    elif self.label_texts_ensemble == 'prompt8':
        f = f'./DeCLIP/prototype/data/datasets/prompts/query_pattern_prompt8'
    elif self.label_texts_ensemble == 'prompt80':
        f = f'./DeCLIP/prototype/data/datasets/prompts/query_pattern_prompt80'
    elif self.label_texts_ensemble == 'cc':
        return [text]
    elif 'file:' in self.label_texts_ensemble:
        f = self.label_texts_ensemble[5:]
    elif self.label_texts_ensemble == 'simple':
        f = f'./DeCLIP/prototype/data/datasets/prompts/query_pattern_prompt1'
    else:
        raise NotImplementedError(self.label_texts_ensemble)
    label_text = []
    with open(f) as fin:
        for line in fin.readlines():
            label_text.append(line.strip().replace('{0}', text))
    return label_text


clip_dataset.ClipDataset._get_label_text = _get_label_text

# DECLIP.forward all_gather
from declip import DECLIP
from prototype.model import declip
declip.DECLIP = DECLIP


# broadcast uint8 --> int32
from prototype.utils.dist import _serialize_to_tensor
import linklink as link
import pickle

def broadcast_object(obj, group=None):
    """make suare obj is picklable
    """
    if link.get_world_size() == 1:
        return obj

    serialized_tensor = _serialize_to_tensor(obj).cuda()

    if serialized_tensor.dtype == torch.uint8:
        serialized_tensor_raw_dtype = serialized_tensor.dtype
        serialized_tensor = serialized_tensor.int()

    numel = torch.IntTensor([serialized_tensor.numel()]).cuda()
    link.broadcast(numel, 0)
    # serialized_tensor from storage is not resizable
    serialized_tensor = serialized_tensor.clone()
    serialized_tensor.resize_(numel)
    link.broadcast(serialized_tensor, 0)

    if serialized_tensor_raw_dtype == torch.uint8:
        serialized_tensor = serialized_tensor.to(serialized_tensor_raw_dtype)

    serialized_bytes = serialized_tensor.cpu().numpy().tobytes()
    deserialized_obj = pickle.loads(serialized_bytes)
    return deserialized_obj


# optim_entry add npu fused optimizers
from prototype.optimizer import optim_entry as optim_entry_raw


def optim_entry(config):
    npu_fused_optims = {
        'Adam': apex.optimizers.NpuFusedAdam,
        'AdamW': apex.optimizers.NpuFusedAdamW,
        'SGD': apex.optimizers.NpuFusedSGD,
    }

    if npu_fused_optims.get(config['type'], False):
        return npu_fused_optims[config['type']](**config['kwargs'])

    return optim_entry_raw(config)
