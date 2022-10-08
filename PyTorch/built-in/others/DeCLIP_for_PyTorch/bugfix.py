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

import os
import torch

if torch.__version__ >= "1.8":
    import torch_npu
import apex


# base_dataset limit world_size up to 8 to avoid eval errors during multi-level multi-card training
from prototype.data.datasets import base_dataset


def merge(self, prefix):
    """
    Merge results into one file.

    Arguments:
        - prefix (:obj:`str`): dir/results.rank
    """
    world_size = min(link.get_world_size(), 8)
    merged_file = prefix.rsplit('.', 1)[0] + '.all'
    merged_fd = open(merged_file, 'w')
    for rank in range(world_size):
        res_file = prefix + str(rank)
        assert os.path.exists(res_file), f'No such file or directory: {res_file}'
        with open(res_file, 'r') as fin:
            for line_idx, line in enumerate(fin):
                merged_fd.write(line)
    merged_fd.close()
    return merged_file


base_dataset.BaseDataset.merge = merge

# _get_label_text ....-->./DeCLIP
from prototype.data.datasets import clip_dataset


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

# makedir
def makedir(path):
    if link.get_rank() % 8 == 0 and not os.path.exists(path):
        os.makedirs(path)
    link.barrier()

# initialize bugfixed when backend='nccl'
import torch.distributed as dist


def initialize(backend='nccl'):
    port = "12345"
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    if backend == 'nccl':
        dist.init_process_group(backend='nccl', rank=proc_id, world_size=ntasks)
    else:
        dist.init_process_group(backend='gloo', rank=proc_id, world_size=ntasks)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)


link.initialize = initialize

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

# DECLIP.forward all_gather
from declip import DECLIP
from prototype.model import declip

declip.DECLIP = DECLIP

